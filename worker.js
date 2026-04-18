import { pipeline, env, AutoProcessor, AutoModelForCausalLM, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@latest';

let embeddingPipeline = null;
let llmProcessor = null;
let llmModel = null;
let isAborted = false;

// Heartbeat para confirmar vida
setInterval(() => {
    self.postMessage({ action: 'heartbeat', payload: { now: Date.now() } });
}, 1000);

self.onmessage = async (e) => {
    const { action, payload, id } = e.data;
    try {
        if (action === 'abort') {
            isAborted = true;
            return;
        }

        if (action === 'reset') {
            embeddingPipeline = null;
            llmProcessor = null;
            llmModel = null;
            self.postMessage({ action: 'status', payload: { text: 'Motor liberado. Listo para reinicio.' } });
            return;
        }

        if (action === 'init') {
            const { modelId } = payload;
            self.postMessage({ action: 'status', payload: { text: 'Configurando Motor WebGPU...' } });
            
            // PERSISTENCIA: Intentamos usar el caché del navegador para no descargar 2GB cada vez
            env.allowLocalModels = false;
            env.useBrowserCache = true; 
            env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/';
            env.backends.onnx.wasm.numThreads = 1;
            
            self.postMessage({ action: 'status', payload: { text: 'Cargando Motor de Vectores (RAG)...' } });
            embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { 
                device: 'webgpu',
                progress_callback: (p) => self.postMessage({ action: 'progress', payload: { ...p, model: 'embedding' } })
            });

            self.postMessage({ action: 'status', payload: { text: 'Cargando Motor Generativo (LLM)...' } });
            const progressCb = (info) => {
                if (info.status === 'progress' || info.progress !== undefined) {
                     self.postMessage({ action: 'progress', payload: { ...info, model: 'llm' } });
                }
            };
            
            if (modelId.includes("Qwen")) {
                llmModel = await pipeline('text-generation', modelId, { 
                    device: 'webgpu',
                    progress_callback: progressCb
                });
                llmProcessor = { isFallback: true, tokenizer: llmModel.tokenizer };
            } else {
                llmProcessor = await AutoProcessor.from_pretrained(modelId);
                llmModel = await AutoModelForCausalLM.from_pretrained(modelId, { 
                    dtype: "q4f16",
                    device: 'webgpu',
                    progress_callback: progressCb
                });
            }

            self.postMessage({ action: 'ready' });
            
        } else if (action === 'embed') {
            const output = await embeddingPipeline(payload.text, { pooling: 'mean', normalize: true });
            self.postMessage({ action: 'embed_result', id, payload: { vector: Array.from(output.data) } });
            
        } else if (action === 'generate') {
            const { sys, prompt } = payload;
            isAborted = false; // Reset on start
            
            const checkAbort = (text) => {
                if(isAborted) throw new Error("Generación detenida por el usuario.");
                self.postMessage({ action: 'chunk', payload: { text } });
            };

            if(llmProcessor && llmProcessor.isFallback) {
                 const messages = [ { role: "system", content: sys }, { role: "user", content: prompt } ];
                 const promptTemplate = llmModel.tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
                 
                 const streamer = new TextStreamer(llmModel.tokenizer, {
                     skip_prompt: true,
                     skip_special_tokens: true,
                     callback_function: checkAbort
                 });
                 
                 await llmModel(promptTemplate, { max_new_tokens: 1536, streamer, do_sample: false });
                 self.postMessage({ action: 'generate_complete' });
            } else {
                 const messages = [ { role: 'system', content: sys }, { role: 'user', content: [{ type: "text", text: prompt }] } ];
                 const promptTemplate = llmProcessor.apply_chat_template(messages, { enable_thinking: false, add_generation_prompt: true });
                 const inputs = await llmProcessor(promptTemplate);
                 
                 const streamer = new TextStreamer(llmProcessor.tokenizer, {
                     skip_prompt: true,
                     skip_special_tokens: true,
                     callback_function: checkAbort
                 });
                 
                 await llmModel.generate({ ...inputs, max_new_tokens: 2048, do_sample: false, streamer });
                 self.postMessage({ action: 'generate_complete' });
            }
        }
    } catch (err) {
        if(err.message === "Generación detenida por el usuario.") {
            self.postMessage({ action: 'status', payload: { text: "Operación cancelada." } });
        }
        self.postMessage({ action: 'error', payload: { message: (err.message || String(err)) } });
    }
};
