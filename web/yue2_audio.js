import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * YuE2 Audio Preview & Saver - Frontend Extension
 * Restores and attaches the interactive audio player widget (play/pause controls,
 * progress scrubber, volume, and download) directly onto the YuE2AudioSavePreview node canvas.
 */

function getAudioUrl(fileInfo) {
    if (!fileInfo || !fileInfo.filename) return "";
    const filename = encodeURIComponent(fileInfo.filename);
    const subfolder = encodeURIComponent(fileInfo.subfolder || "");
    const type = encodeURIComponent(fileInfo.type || "output");
    const rand = app.getRandParam ? app.getRandParam() : `&rand=${Date.now()}`;
    return api.apiURL(`/view?filename=${filename}&type=${type}&subfolder=${subfolder}${rand}`);
}

function updateAudioUI(node, audioUrl) {
    if (!node || !audioUrl) return;

    const widget = node.widgets?.find(w => w.name === "audioUI");
    if (widget) {
        widget.value = audioUrl;

        // Update Pinia / Vue reactive state if present
        if (widget.options?.setValue) {
            try {
                widget.options.setValue(audioUrl);
            } catch (e) {
                console.warn("[YuE2 Audio Preview] Failed to set widget value via options:", e);
            }
        }

        // Update classic DOM audio element
        if (widget.element) {
            widget.element.src = audioUrl;
            widget.element.classList?.remove("empty-audio-widget");
            try {
                widget.element.load();
            } catch (e) {}
        }

        // Invoke widget callback
        if (typeof widget.callback === "function") {
            try {
                widget.callback(audioUrl);
            } catch (e) {}
        }
    }

    if (app.graph) {
        app.graph.setDirtyCanvas(true, true);
    }
}

app.registerExtension({
    name: "YuE2.AudioPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YuE2AudioSavePreview") {
            // Append audioUI into optional inputs so ComfyUI's native AudioWidget
            // (Vue AudioPreviewPlayer in modern ComfyUI or HTML5 <audio> in classic)
            // attaches to the node without shifting the order of existing widgets.
            nodeData.input = nodeData.input || {};
            nodeData.input.optional = nodeData.input.optional || {};
            nodeData.input.optional.audioUI = ["AUDIO_UI", {}];

            // Wrap onNodeCreated for layout sizing and fallback DOM widget creation
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const res = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

                // Ensure node has enough space for playback controls
                const minWidth = 380;
                const minHeight = 240;
                if (this.size[0] < minWidth) this.size[0] = minWidth;
                if (this.size[1] < minHeight) this.size[1] = Math.max(this.size[1], minHeight);

                // Robust Fallback: If AUDIO_UI was not instantiated by ComfyUI, create DOM audio element
                setTimeout(() => {
                    let audioWidget = this.widgets?.find(w => w.name === "audioUI");
                    if (!audioWidget) {
                        const audioEl = document.createElement("audio");
                        audioEl.controls = true;
                        audioEl.className = "comfy-audio";
                        audioEl.setAttribute("name", "media");
                        audioEl.style.width = "100%";
                        audioEl.style.height = "42px";
                        audioEl.style.marginTop = "6px";
                        audioEl.style.boxSizing = "border-box";

                        audioWidget = this.addDOMWidget("audioUI", "audioUI", audioEl, {
                            serialize: false,
                            getValue: () => audioEl.src || "",
                            setValue: (v) => {
                                audioEl.src = v || "";
                                if (v) audioEl.classList.remove("empty-audio-widget");
                            }
                        });
                        audioWidget.serialize = false;
                        if (audioWidget.options) audioWidget.options.serialize = false;
                        if (app.graph) app.graph.setDirtyCanvas(true, true);
                    }
                }, 20);

                return res;
            };

            // Wrap onExecuted to update preview URL as soon as execution completes
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                origOnExecuted?.apply(this, arguments);

                if (message?.audio && message.audio.length > 0) {
                    const audioUrl = getAudioUrl(message.audio[0]);
                    updateAudioUI(this, audioUrl);
                }
            };
        }
    },

    onNodeOutputsUpdated(outputMap) {
        if (!outputMap) return;

        for (const [nodeId, output] of Object.entries(outputMap)) {
            if (!output?.audio?.length) continue;

            const graph = app.graph || app.rootGraph;
            const node = graph?.getNodeById ? graph.getNodeById(nodeId) : null;
            if (!node || (node.type !== "YuE2AudioSavePreview" && node.comfyClass !== "YuE2AudioSavePreview")) continue;

            const audioUrl = getAudioUrl(output.audio[0]);
            updateAudioUI(node, audioUrl);
        }
    }
});
