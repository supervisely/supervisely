Vue.component("sly-flow-diagram", {
  template: ``,
  props: {
    connections: {
      type: Array,
      default: () => {},
    },
  },
  methods: {
    addArc(elmsPath, radius) {
      const reL = /^L ?([\d.\-+]+) ([\d.\-+]+) ?/;
      let newXY;

      function getDir(xy1, xy2) {
        if (xy1.x === xy2.x) {
          return xy1.y < xy2.y ? "d" : "u";
        } else if (xy1.y === xy2.y) {
          return xy1.x < xy2.x ? "r" : "l";
        } else if (Math.abs(xy1.x - xy2.x) < 1) {
          return xy1.y < xy2.y ? "d" : "u";
        } else if (Math.abs(xy1.y - xy2.y) < 1) {
          return xy1.x < xy2.x ? "r" : "l";
        }
        // throw new Error("Invalid data");
      }

      function captureXY(s, x, y) {
        newXY = { x: +x, y: +y };
        return "";
      }

      function offsetXY(xy, dir, offsetLen, toBack) {
        return {
          x:
            xy.x +
            (dir === "l" ? -offsetLen : dir === "r" ? offsetLen : 0) *
              (toBack ? -1 : 1),
          y:
            xy.y +
            (dir === "u" ? -offsetLen : dir === "d" ? offsetLen : 0) *
              (toBack ? -1 : 1),
        };
      }

      let curXY,
        pathData = elmsPath
          .getAttribute("d")
          .trim()
          .replace(/,/g, " ")
          .replace(/\s+/g, " ")
          .replace(/^M ?([\d.\-+]+) ([\d.\-+]+) ?/, (s, x, y) => {
            curXY = { x: +x, y: +y };
            return "";
          });
      if (!curXY) {
        throw new Error("Invalid data");
      }
      let newPathData = "M" + curXY.x + " " + curXY.y;

      let curDir;
      //   console.log("pathData", pathData);
      while (pathData) {
        // newXY = null;
        pathData = pathData.replace(reL, captureXY);
        if (!newXY) {
          throw new Error("Invalid data");
        }

        const newDir = getDir(curXY, newXY);

        if (curDir) {
          const arcStartXY = offsetXY(curXY, curDir, radius, true),
            arcXY = offsetXY(curXY, newDir, radius),
            sweepFlag =
              curDir === "l" && newDir === "u"
                ? "1"
                : curDir === "l" && newDir === "d"
                ? "0"
                : curDir === "r" && newDir === "u"
                ? "0"
                : curDir === "r" && newDir === "d"
                ? "1"
                : curDir === "u" && newDir === "l"
                ? "0"
                : curDir === "u" && newDir === "r"
                ? "1"
                : curDir === "d" && newDir === "l"
                ? "1"
                : curDir === "d" && newDir === "r"
                ? "0"
                : null;
          if (!sweepFlag) {
            throw new Error("Invalid data");
          }
          newPathData +=
            "L" +
            arcStartXY.x +
            " " +
            arcStartXY.y +
            "A " +
            radius +
            " " +
            radius +
            " 0 0 " +
            sweepFlag +
            " " +
            arcXY.x +
            " " +
            arcXY.y;
        }

        curXY = newXY;
        curDir = newDir;
      }
      newPathData += "L" + curXY.x + " " + curXY.y;
      //   console.log("newPathData", newPathData);
      elmsPath.setAttribute("d", newPathData);
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
        // const SELECTOR = ".leader-line:last-child .leader-line-line-path";
        // this.addArc(document.querySelector(SELECTOR), 10);
      });
    }
    setTimeout(() => {
      console.log("Timeout");
      document.querySelectorAll(".leader-line-line-path").forEach((line) => {
        try {
          this.addArc(line, 10);
        } catch (error) {}
      });
    }, 3500);
    // documents
    console.log("Connections drawn");
  },
  updated() {
    console.log("Flow diagram updated");
  },
});
