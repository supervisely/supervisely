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
      const target = connections[sourceId];

      target.forEach((targetData) => {
        console.log(targetData);

        // window.arrowLine(`#node-${sourceId}`, `#node-${targetId}`, {
        //   color: "#2ecc71",
        //   thickness: 3,
        //   endPlugSize: 2,
        //     style: "dot", // dash, dot-dash, solid
        //     endPoint: {type: "circles    ", size: 5, position: "both},
        //   }
        line = new LeaderLine(
          document.getElementById(sourceId),
          document.getElementById(targetData[0]),
          { ...targetData[1] }
        );
        middleLabel = targetData[1].middleLabel;
        fontSize = targetData[1].fontSize;
        fontColor = targetData[1].fontColor;
        fontFamily = targetData[1].fontFamily;
        if (targetData[1].labelType === "path") {
          line.middleLabel = LeaderLine.pathLabel(middleLabel, {
            color: fontColor,
            fontSize: fontSize,
            fontFamily: fontFamily,
          });
        } else {
          line.middleLabel = LeaderLine.captionLabel(middleLabel, {
            color: fontColor,
            fontSize: fontSize,
            fontFamily: fontFamily,
          });
        }
        pointAnchor = targetData[1].pointAnchor;
        if (pointAnchor) {
          console.log("pointAnchor", pointAnchor);
          line.end = LeaderLine.pointAnchor(
            document.getElementById(targetData[0]),
            pointAnchor
          );
        }
        console.log(
          "Connection drawn from",
          sourceId,
          "to",
          targetData[0],
          "options",
          targetData[1]
        );
      });
    }
    // });
    console.log("Connections drawn");
  },
  updated() {
    console.log("Flow diagram updated");
  },
});
