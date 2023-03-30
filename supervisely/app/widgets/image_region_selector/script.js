const POINT_SIZE = 6;
const VIEW_BOX_OFFSET = 60;
const VIEW_BOX_OFFSET_HALF = VIEW_BOX_OFFSET / 2;

function getViewBox(viewBox) {
  viewBox.height += VIEW_BOX_OFFSET;
  viewBox.h += VIEW_BOX_OFFSET;
  viewBox.width += VIEW_BOX_OFFSET;
  viewBox.w += VIEW_BOX_OFFSET;
  viewBox.x -= VIEW_BOX_OFFSET_HALF;
  viewBox.x2 += VIEW_BOX_OFFSET_HALF;
  viewBox.y -= VIEW_BOX_OFFSET_HALF;
  viewBox.y2 += VIEW_BOX_OFFSET_HALF;

  return viewBox;
}

function getBBoxSize(bbox) {
  return {
    width: bbox[1][0] - bbox[0][0],
    height: bbox[1][1] - bbox[0][1],
  };
}

Vue.component('smarttool-editor', {
  template: `
    <div v-loading="loading" style="position: relative;">
      <div v-if="disabled" style="position: absolute; inset: 0; opacity: 0.5; background-color: #808080;"></div>
      <svg ref="container" xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%"></svg>
    </div>
  `,
  props: {
    maskOpacity: 0.5,
    bbox: {
      type: Array,
      required: true,
    },
    imageUrl: {
      type: String,
      required: true,
    },
    disabled: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      pt: null,
      container: null,
      loading: true,
      contours: [],
    };
  },
  watch: {
    imageUrl() {
      this.group.clear()
      const viewBox = getViewBox(this.bboxEl.bbox());
      this.sceneEl.viewbox(viewBox)
      this.backgroundEl = this.sceneEl.image(this.imageUrl).loaded(() => {
        this.pointSize = POINT_SIZE * (viewBox.w / this.container.width.baseVal.value);
        this.initPoints();
      });
      this.group.add(
        this.backgroundEl,
        this.bboxEl
      );
    },
    bbox() {
      const bboxSize = getBBoxSize(this.bbox);
      this.bboxEl.size(bboxSize.width, bboxSize.height)
        .move(this.bbox[0][0], this.bbox[0][1])

      this.sceneEl.viewbox(getViewBox(this.bboxEl.bbox()))
    },
  },
  methods: {
    initPoints() {
      this.bboxEl.node.nextElementSibling.childNodes.forEach((n) => {
        if (!n.r) return;
        n.setAttribute('r', this.pointSize);
      });

      this.loading = false;
    },
    init() {
      this.container.addEventListener('contextmenu', (e) => {
        e.preventDefault();
      });

      this.sceneEl = SVG(this.container)
        .panZoom({
          zoomMin: 0.1,
          zoomMax: 20,
          panButton: 2
        });

      this.group = this.sceneEl.group();

      const bboxSize = getBBoxSize(this.bbox);

      this.bboxEl = this.sceneEl
        .rect(bboxSize.width, bboxSize.height)
        .move(this.bbox[0][0], this.bbox[0][1])
        .selectize()
        .resize()
        .attr({
          "fill-opacity": 0,
        })
        .on('resizedone', () => {
          let x = this.bboxEl.x();
          let y = this.bboxEl.y();
          let w = this.bboxEl.width();
          let h = this.bboxEl.height();
          let image_width = this.backgroundEl.node.width.baseVal.value;
          let image_height = this.backgroundEl.node.height.baseVal.value;

          if (x < 0) { x = 0 }
          if (y < 0) { y = 0 }
          if ((x + w) > image_width) { w = image_width - x}
          if ((y + h) > image_height) { h = image_height - y}
          this.$emit('update:bbox', [[x, y], [x + w, y + h]]);
        });

      const viewBox = getViewBox(this.bboxEl.bbox());
      this.sceneEl.viewbox(viewBox)

      this.backgroundEl = this.sceneEl.image(this.imageUrl).loaded(() => {
        this.pointSize = POINT_SIZE * (viewBox.w / this.container.width.baseVal.value);
        this.initPoints();
      });
      this.group.add(
        this.backgroundEl,
        this.bboxEl
      );

      this.pt = this.container.createSVGPoint();
    },
  },

  mounted() {
    this.pointsMap = new Map();
    this.container = this.$refs['container'];

    this.init();
  }
});