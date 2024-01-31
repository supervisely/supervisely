Vue.component("video-tracking-gates", {
  props: {
    brushSize: {
      type: Number,
      default: 12,
    },
    width: {
      type: Number,
      //   default: 640,
    },
    height: {
      type: Number,
      //   default: 480,
    },
    outputName: {
      type: String,
      default: "canvas",
    },
  },
  template: `
    <div
        class="canvas-wrapper"
        ref="canvasWrapper" 
        :style="overlayStyle"
    >
    <div class="draw-area">
      <canvas id="canvas" ref="canvas" :width="width" :height="height"></canvas>
      <canvas id="cursor" ref="cursor" :width="width" :height="height"></canvas>
    </div>
    <ul class="tools">
      <li id="tool-pencil" :class="{ active: selectedToolIdx === 0 }" @click="changeTool(0)">
        <span>brush</span>
        </li>
        <li id="tool-eraser" :class="{ active: selectedToolIdx === 1 }" @click="changeTool(1)">
        <span>eraser</span>
        </li>
        <li id="tool-color-palette" @click="showColorPalette()">
        <span>palette</span>
        </li>
        <li id="tool-download" @click="download()">
       <span>download</span>
      </li>
    </ul>
  </div>
    `,
  data() {
    return {
      canvasContext: null,
      cursorContext: null,
      isDrawing: false,
      lastX: 0,
      lastY: 0,
      firstX: 0,
      firstY: 0,
      tools: [
        {
          name: "Pencil",
          color: "#000000",
        },
        {
          name: "Eraser",
        },
      ],
      selectedToolIdx: 0,
      lines: [],
      lineStart: [],
      lineEnd: [],
    };
  },
  computed: {
    overlayStyle() {
      return {
        width: `${this.width}px`,
        height: `${this.height}px`,
        opacity: 0.8,
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundSize: "contain",
      };
    },
  },
  mounted() {
    this.setCanvas();
    this.bindEvents();
  },
  methods: {
    getCanvasSize() {
      // #TODO: get canvas size from parent
    },
    setCanvas() {
      this.$refs[
        "canvasWrapper"
      ].style.gridTemplateColumns = `${this.width}px 30px`;
      this.$refs["canvasWrapper"].style.width = `${this.width + 30}px`;
      this.$refs["canvasWrapper"].style.height = `${this.height}px`;

      this.canvasContext = this.$refs["canvas"].getContext("2d");
      this.canvasContext.lineJoin = "round";
      this.canvasContext.lineCap = "round";
      this.canvasContext.lineWidth = this.brushSize;
      this.canvasContext.strokeStyle = this.tools[this.selectedToolIdx].color;

      this.cursorContext = this.$refs["cursor"].getContext("2d");
    },
    bindEvents() {
      this.$refs["canvas"].addEventListener("mousedown", (event) => {
        this.isDrawing = true;
        [this.firstX, this.firstY] = [event.offsetX, event.offsetY];
        [this.lastX, this.lastY] = [event.offsetX, event.offsetY];
        this.updateCurrentLine(event, "start");
        // this.clean_canvas();
      });
      //   this.$refs["canvas"].addEventListener("mousemove", this.draw);
      this.$refs["canvas"].addEventListener("mousemove", (event) => {
        if (!this.isDrawing) return;
        this.updateCurrentLine(event, "end");
        // console.log(`start: ${this.lineStart}, end: ${this.lineEnd}`);
        // this.clean_canvas();
        this.drawCurrentLine();
        this.draw(event);
        setTimeout(() => {
          this.clean_canvas();
        }, 100);
      });
      //   this.$refs["canvas"].addEventListener("mouseup", this.draw);
      this.$refs["canvas"].addEventListener("mouseup", (event) => {
        this.updateCurrentLine(event, "end");
        this.lines.push({ start: this.lineStart, end: this.lineEnd });
        // this.clean_canvas();
        this.draw(event);
        // console.log(this.lines);
        this.isDrawing = false;
      });
      //   this.$refs["canvas"].addEventListener("mouseout", (event) => {
      //     this.draw(event);
      //     this.isDrawing = false;
      //   });
    },
    changeTool(tool) {
      this.selectedToolIdx = tool;
      if (tool === 0) {
        this.canvasContext.lineWidth = this.brushSize;
      } else {
        this.canvasContext.lineWidth = this.brushSize * 3;
      }
    },
    updateCurrentLine(event, side) {
      if (this.tools[this.selectedToolIdx].name !== "Eraser") {
        if (side === "start") {
          this.lineStart = [event.offsetX, event.offsetY];
        } else {
          this.lineEnd = [event.offsetX, event.offsetY];
        }
      }
    },
    clean_canvas() {
      // this.canvasContext.beginPath();
      this.canvasContext.clearRect(0, 0, this.width, this.height);
    },
    drawCurrentLine() {
      this.canvasContext.beginPath();
      this.canvasContext.moveTo(this.lineStart[0], this.lineStart[1]);
      this.canvasContext.lineTo(this.lineEnd[0], this.lineEnd[1]);
      this.canvasContext.stroke();
    },
    draw(event) {
      if (!this.isDrawing) return;
      this.drawCursor(event);
      console.log(`lines: ${this.lines}`);
      if (this.lines.length < 2) return;
      for (let i = 0; i < this.lines.length; i++) {
        let line = this.lines[i];
        console.log(`line: ${line}`);
        this.canvasContext.beginPath();
        this.canvasContext.moveTo(line.start[0], line.start[1]);
        this.canvasContext.lineTo(line.end[0], line.end[1]);
        this.canvasContext.stroke();
      }
      //   if (this.tools[this.selectedToolIdx].name === "Eraser") {
      //     this.canvasContext.globalCompositeOperation = "destination-out";
      //     fromX = this.lastX;
      //     fromY = this.lastY;
      //   } else {
      //     this.canvasContext.globalCompositeOperation = "source-over";
      //     this.canvasContext.strokeStyle = this.tools[this.selectedToolIdx].color;
      //     fromX = this.firstX;
      //     fromY = this.firstY;
      //   }

      //   this.canvasContext.beginPath();
      //   this.canvasContext.moveTo(fromX, fromY);
      //   this.canvasContext.lineTo(event.offsetX, event.offsetY);
      //   this.canvasContext.stroke();
      //   [this.lastX, this.lastY] = [event.offsetX, event.offsetY];
    },
    // tempDraw(event) {
    //   if (!this.isDrawing) return;
    //   this.draw(event);
    //   setTimeout(() => {
    //     if (this.tools[this.selectedToolIdx].name !== "Eraser") {
    //       this.canvasContext.globalCompositeOperation = "destination-out";
    //       this.canvasContext.lineWidth = this.brushSize + 1;
    //       this.canvasContext.beginPath();
    //       this.canvasContext.moveTo(this.firstX, this.firstY);
    //       this.canvasContext.lineTo(event.offsetX, event.offsetY);
    //       this.canvasContext.stroke();
    //       this.canvasContext.lineWidth = this.brushSize;
    //     }
    //     // this.cursorContext.clearRect(0, 0, this.width, this.height);
    //   }, 10);
    // },
    // tempDraw(event) {
    //   this.drawCursor(event, this.brushSize * 10);
    //   if (!this.isDrawing) return;

    //   //   this.canvasContext.globalCompositeOperation = "destination-out";
    //   if (this.tools[this.selectedToolIdx].name === "Eraser") {
    //     this.canvasContext.beginPath();
    //     this.canvasContext.globalCompositeOperation = "destination-out";
    //     this.canvasContext.moveTo(this.lastX, this.lastY);
    //     this.canvasContext.lineTo(event.offsetX, event.offsetY);
    //     this.canvasContext.stroke();
    //   } else {
    //     this.canvasContext.strokeStyle = this.tools[this.selectedToolIdx].color;
    //     this.canvasContext.globalCompositeOperation = "source-over";
    //     this.canvasContext.beginPath();
    //     this.canvasContext.moveTo(this.firstX, this.firstY);
    //     this.canvasContext.lineTo(event.offsetX, event.offsetY);
    //     this.canvasContext.stroke();
    //   }
    //   [this.lastX, this.lastY] = [event.offsetX, event.offsetY];
    // },
    drawCursor(event) {
      this.cursorContext.beginPath();
      this.cursorContext.ellipse(
        event.offsetX,
        event.offsetY,
        this.brushSize,
        this.brushSize,
        Math.PI / 4,
        0,
        2 * Math.PI
      );
      this.cursorContext.stroke();
      setTimeout(() => {
        this.cursorContext.clearRect(0, 0, this.width, this.height);
      }, 100);
    },
    showColorPalette() {
      const colorPalette = document.createElement("input");
      colorPalette.addEventListener("change", (event) => {
        this.tools[0].color = event.target.value;
      });
      colorPalette.type = "color";
      colorPalette.value = this.tools[0].color;
      colorPalette.click();
    },
    download() {
      const link = document.createElement("a");
      link.download = `${this.outputName}.png`;
      link.href = this.$refs["canvas"].toDataURL();
      link.click();
    },
  },
});
