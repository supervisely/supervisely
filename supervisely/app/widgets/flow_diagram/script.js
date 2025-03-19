Vue.component("sly-flow-diagram", {
  template: ``,
  props: {
    connections: {
      type: Array,
      default: () => {},
    },
  },
  mounted() {
    console.log("Flow diagram mounted");
    const connections = this.connections;
    for (const sourceId in connections) {
      const targetIds = connections[sourceId];
      targetIds.forEach((targetId) => {
        window.arrowLine(`#node-${sourceId}`, `#node-${targetId}`);
        console.log("Connection drawn from", sourceId, "to", targetId);
      });
    }
    console.log("Connections drawn");
  },
  updated() {
    console.log("Flow diagram updated");
  },
});
