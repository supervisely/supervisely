Vue.component('heatmap-image', {
  template: `
  <div
    @click="$emit('click')"
  >
    <div class="heatmap-header" :style="widthStyle">
      <div class="opacity-slider" @click.stop>
        <div class="opacity-label">
          <span class="opacity-label-text">Opacity:</span>
          <span class="opacity-value">{{opacity}}%</span>
        </div>
        
        <el-slider
          type="range"
          :min="0"
          :max="100"
          :step="1"
          :value="opacity"
          @input="$emit('update:opacity', $event)"
          class="slider"
        >
      </div>
      
      <div class="legend" @click.stop>
        <span class="legend-label legend-min">{{ formatValue(minValue) }}</span>
        <div class="legend-gradient" :style="{ background: gradientStyle }"></div>
        <span class="legend-label legend-max">{{ formatValue(maxValue) }}</span>
      </div>
    </div>

    <div
      class="image-wrapper"
      ref="wrapper"
      :style="imageWrapperStyle"
      @click.stop="handleImageClick"
    >
      <img
      class="base-image"
      :src="backgroundUrl"
      @load="handleImageLoad"
      >
      <img
      class="overlay-image"
      :style="{ opacity: opacity / 100 }"
      :src="maskUrl"
      >
      
      <div 
        v-if="clickedValue !== null" 
        class="click-indicator"
        :style="indicatorStyle"
      >
        <div class="click-dot"></div>
      </div>
      
      </div>
    <div 
      v-if="clickedValue !== null" 
      class="value-popup"
      :style="popupStyle"
    >
      <div class="value-popup-content">
        <span class="value-popup-value">{{ formatValue(clickedValue) }}</span>
      </div>
      <div class="value-popup-arrow"></div>
    </div>
  </div>
  `,
  data() {
    return {
      naturalWidth: null,
      naturalHeight: null,
      clickX: 0,
      clickY: 0,
      popupX: 0,
      popupY: 0,
    };
  },
  computed: {
    gradientStyle() {
      return `linear-gradient(to right, ${this.legendColors.join(', ')})`;
    },
    widthStyle() {
      const styles = {};
      
      if (this.width) {
        styles.width = typeof this.width === 'number' ? `${this.width}px` : this.width;
      } else if (this.height && this.naturalWidth && this.naturalHeight) {
        const heightValue = typeof this.height === 'number' ? this.height : parseFloat(this.height);
        const aspectRatio = this.naturalWidth / this.naturalHeight;
        styles.width = `${heightValue * aspectRatio}px`;
      }
      
      return styles;
    },
    imageWrapperStyle() {
      const styles = { ...this.widthStyle };
      
      if (this.naturalWidth && this.naturalHeight) {
        styles.aspectRatio = `${this.naturalWidth} / ${this.naturalHeight}`;
      }
      
      return styles;
    },
    indicatorStyle() {
      return {
        left: `${this.clickX}px`,
        top: `${this.clickY}px`
      };
    },
    popupStyle() {
      return {
        left: `${this.popupX}px`,
        top: `${this.popupY}px`
      };
    },
  },
  methods: {
    handleImageLoad(event) {
      this.naturalWidth = event.target.naturalWidth;
      this.naturalHeight = event.target.naturalHeight;
    },
    handleImageClick(event) {
      if (!this.maskData?.length) return;

      const wrapper = this.$refs.wrapper;
      const wrapperRect = wrapper.getBoundingClientRect();

      const wrapperX = event.clientX - wrapperRect.left;
      const wrapperY = event.clientY - wrapperRect.top;
      this.clickX = wrapperX;
      this.clickY = wrapperY;

      const maskH = this.maskData.length;
      const maskW = this.maskData[0].length;
      let maskX = Math.floor(wrapperX * (maskW / wrapperRect.width));
      let maskY = Math.floor(wrapperY * (maskH / wrapperRect.height));
      maskX = Math.min(Math.max(maskX, 0), maskW-1)
      maskY = Math.min(Math.max(maskY, 0), maskH-1)

      console.log('emiting update:mask-x')
      this.$emit('update:mask-x', maskX);
      this.$emit('update:mask-y', maskY);
      this.$emit('update:clicked-value', this.maskData[maskY][maskX]);

      this.popupX = event.clientX;
      this.popupY = event.clientY;

      if (this.onImageClick) {
        this.onImageClick();
      }
    },
    formatValue(value) {
      if (value === null || value === undefined) return 'N/A';
      if (Number.isInteger(value)) {
        return value.toString();
      }
      const abs = Math.abs(value);
      let decimals;
      if (abs >= 1000) decimals = 1;
      else if (abs >= 100) decimals = 2;
      else if (abs >= 1) decimals = 3;
      else if (abs >= 0.01) decimals = 4;
      else decimals = 5;

      return parseFloat(value.toFixed(decimals)).toString();
    }
  },
  watch: {
    backgroundUrl() {
      this.naturalWidth = null;
      this.naturalHeight = null;
    }
  },
  props: {
    backgroundUrl: String,
    maskUrl: String,
    maskData: Array,
    opacity: Number,
    width: [Number, String],
    height: [Number, String],
    legendColors: {
      type: Array,
      default: () => ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    },
    minValue: Number,
    maxValue: Number,
    clickedValue: Number,
    maskX: Number,
    maskY: Number,
    onImageClick: {
      type: Function,
      default: null
    }
  },
});
