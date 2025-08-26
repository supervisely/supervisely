Vue.component("sly-flow", {
  props: ["id", "state", "data", "context", "post", "session", "url"],
  data: function () {
    return {
      componentKey: 0,
    };
  },
  methods: {
    sendMessageToFrame(params) {
      console.log("+++ sendMessageToFrame", params);
      this.$refs.frame.contentWindow.postMessage({
        action: params.action,
        payload: params.data,
      });
    },

    onFrameMessage(e) {
      console.log("+++ iframe message received. e:", e);
      let data = e.data;

      if (typeof data === "string") {
        data = JSON.parse(e.data);
      }

      console.log("+++ iframe message received:", data);

      if (data.action === "node-clicked") {
        const { nodeId, link } = data.payload;

        if (link?.url) {
          window.open(link.url, "_blank");
        } else if (link?.action) {
          this.post(link.action);
        } else {
          this.post(`/${nodeId}/node_clicked_cb`);
        }
      } else if (data.action === "loaded") {
        console.log("+++", this.state[this.id].nodes);
        // if (this.$refs.frame.contentWindow.slyApp) {
        console.log("+++111", this.state[this.id].nodes);
        this.$refs.frame.contentWindow.slyApp.flow = {
          nodes: this.state[this.id].nodes,
          edges: this.state[this.id].edges,
          sidebarNodes: this.state[this.id].sidebarNodes,
        };
        // this.refreshNodes();
        this.sendMessageToFrame({
          action: "flow-refresh",
          data: {},
        });
        this.sendMessageToFrame({
          action: "sidebar-nodes-refresh",
          data: {},
        });
      } else if (data.action === "node-updated") {
        this.post(`/${this.id}/node_updated_cb`, data.payload);
      } else if (data.action === "edge-created") {
        console.log("+++ edge-created", data.payload);
        this.post(`/${this.id}/edge_added_cb`, data.payload);
      } else if (data.action === "node-added") {
        this.post(`/${this.id}/node_added_cb`, data.payload);
      }
    },
  },

  mounted() {
    console.log("+++ mounted:  nodes", [...this.state[this.id].nodes]);
    window.addEventListener("message", this.onFrameMessage);
    this.$eventBus.$on(`sly-flow-${this.id}`, this.sendMessageToFrame);
  },

  beforeDestroy() {
    console.log("+++ beforeDestroy");
    window.removeEventListener("message", this.onFrameMessage);
    this.$eventBus.$off(`sly-flow-${this.id}`, this.sendMessageToFrame);
  },

  template: `
  <div>
    <iframe ref="frame" :src="url" style="width: 100%; height: 800px; border: none;"></iframe>
  </div>
`,
});
