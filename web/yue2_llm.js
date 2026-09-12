import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * YuE2 LLM Co-Producer & Polisher - Dynamic Model Selector Extension
 * Synchronizes the model dropdown with the selected provider, dynamically fetching
 * live models from LM Studio, Ollama, or provider APIs without displaying a giant mixed list.
 */

async function fetchModelsForNode(node, isManualRefresh = false) {
    if (!node || !node.widgets) return;

    const providerWidget = node.widgets.find(w => w.name === "provider");
    const modelWidget = node.widgets.find(w => w.name === "model");
    const baseUrlWidget = node.widgets.find(w => w.name === "base_url");
    const apiKeyWidget = node.widgets.find(w => w.name === "api_key");
    const refreshBtn = node.widgets.find(w => w.name === "refresh_models_btn");

    if (!providerWidget || !modelWidget) return;

    const provider = providerWidget.value || "LMStudio";
    const baseUrl = baseUrlWidget?.value || "";
    const apiKey = apiKeyWidget?.value || "";

    if (refreshBtn) {
        refreshBtn.label = "⏳ Fetching...";
        if (app.graph) app.graph.setDirtyCanvas(true, true);
    }

    try {
        const query = new URLSearchParams({
            provider: provider,
            base_url: baseUrl,
            api_key: apiKey
        });

        const resp = await api.fetchApi(`/yue2/models?${query.toString()}`);
        if (!resp.ok) {
            if (refreshBtn) refreshBtn.label = "🔄 Refresh Models";
            return;
        }

        const data = await resp.json();
        if (data && Array.isArray(data.models) && data.models.length > 0) {
            const currentVal = modelWidget.value;
            modelWidget.options = modelWidget.options || {};
            modelWidget.options.values = data.models;

            if (data.models.includes(currentVal)) {
                modelWidget.value = currentVal;
            } else if (!isManualRefresh && currentVal) {
                // If loaded from a saved graph, prepend currentVal so the user's saved workflow remains intact
                modelWidget.options.values = [currentVal, ...data.models.filter(m => m !== currentVal)];
                modelWidget.value = currentVal;
            } else {
                modelWidget.value = data.models[0];
            }

            if (refreshBtn) {
                const count = data.models.length;
                const status = data.live ? "Live" : "Presets";
                refreshBtn.label = `🔄 Refresh Models (${count} ${status})`;
            }

            if (app.graph) {
                app.graph.setDirtyCanvas(true, true);
            }
        }
    } catch (e) {
        console.warn("[YuE2 LLM Producer] Could not fetch models for " + provider, e);
        if (refreshBtn) refreshBtn.label = "🔄 Refresh Models";
    }
}

function attachLLMWidgetHooks(node) {
    if (!node || !node.widgets || node._yue2_llm_attached) return;
    node._yue2_llm_attached = true;

    const providerWidget = node.widgets.find(w => w.name === "provider");
    const baseUrlWidget = node.widgets.find(w => w.name === "base_url");
    const apiKeyWidget = node.widgets.find(w => w.name === "api_key");

    if (providerWidget) {
        const origCb = providerWidget.callback;
        providerWidget.callback = function(val) {
            if (origCb) origCb.apply(this, arguments);
            fetchModelsForNode(node, true);
        };
    }

    if (baseUrlWidget) {
        const origBaseUrlCb = baseUrlWidget.callback;
        baseUrlWidget.callback = function(val) {
            if (origBaseUrlCb) origBaseUrlCb.apply(this, arguments);
            fetchModelsForNode(node, true);
        };
    }

    if (apiKeyWidget) {
        const origApiKeyCb = apiKeyWidget.callback;
        apiKeyWidget.callback = function(val) {
            if (origApiKeyCb) origApiKeyCb.apply(this, arguments);
            fetchModelsForNode(node, true);
        };
    }

    // Add a dedicated Refresh Models button if not already present
    let refreshBtn = node.widgets.find(w => w.name === "refresh_models_btn");
    if (!refreshBtn) {
        // Position the button right below the model widget if possible
        const modelIdx = node.widgets.findIndex(w => w.name === "model");
        refreshBtn = node.addWidget("button", "🔄 Refresh Models", null, () => {
            fetchModelsForNode(node, true);
        });
        refreshBtn.name = "refresh_models_btn";
        refreshBtn.serialize = false;

        if (modelIdx !== -1 && node.widgets.length > modelIdx + 1) {
            // Reorder to place right beneath the model widget
            const btn = node.widgets.pop();
            node.widgets.splice(modelIdx + 1, 0, btn);
        }
    }

    // Initial fetch to populate live models for currently selected provider
    setTimeout(() => {
        fetchModelsForNode(node, false);
    }, 150);
}

app.registerExtension({
    name: "YuE2.LLMProducer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YuE2LLMProducer") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const res = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
                attachLLMWidgetHooks(this);
                return res;
            };

            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function() {
                const res = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                attachLLMWidgetHooks(this);
                return res;
            };
        }
    }
});
