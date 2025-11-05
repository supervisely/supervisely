Vue.component('heatmap-image', {
  template: `
  <div
    class="heatmap-container"
    :style="widthStyle"
    @click="$emit('click')"
  >
    <div class="heatmap-header">
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
      @mouseleave="handleMouseLeave"
    >
      <img
      class="base-image"
      :src="backgroundUrl"
      @load="handleImageLoad"
      draggable="false"
      >
      <img
      class="overlay-image"
      :style="{ opacity: opacity / 100 }"
      :src="maskUrl"
      draggable="false"
      >
      
      <div 
        v-if="clickedValue !== null" 
        class="click-indicator"
        :class="{ 'hiding': isHiding }"
        :style="indicatorStyle"
      >
        <div class="click-dot"></div>
      </div>
      
      <div 
        v-if="clickedValue !== null" 
        class="value-popup"
        :class="['popup-position-' + popupPosition, { 'hiding': isHiding }]"
        :style="popupStyle"
      >
        <div class="value-popup-content">
          <span class="value-popup-value">{{ formatValue(clickedValue) }}</span>
        </div>
        <div class="value-popup-arrow"></div>
      </div>
      
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
      popupPosition: 'top', // 'top', 'bottom', 'right', 'left'
      isHiding: false, // Flag for fade-out animation
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
      } else if (this.height) {
        // Use naturalWidth/Height from loaded image, or fallback to maskWidth/Height
        const effectiveWidth = this.naturalWidth || this.maskWidth;
        const effectiveHeight = this.naturalHeight || this.maskHeight;
        
        if (effectiveWidth && effectiveHeight) {
          const heightValue = typeof this.height === 'number' ? this.height : parseFloat(this.height);
          const aspectRatio = effectiveWidth / effectiveHeight;
          styles.width = `${heightValue * aspectRatio}px`;
        }
      }
      
      return styles;
    },
    imageWrapperStyle() {
      const styles = { ...this.widthStyle };
      
      // Use max-height instead of height to allow responsive scaling
      if (this.height) {
        styles.maxHeight = typeof this.height === 'number' ? `${this.height}px` : this.height;
      }
      
      // Use naturalWidth/Height from loaded image, or fallback to maskWidth/Height
      const effectiveWidth = this.naturalWidth || this.maskWidth;
      const effectiveHeight = this.naturalHeight || this.maskHeight;
      
      // Always use aspect-ratio if we have dimensions
      if (effectiveWidth && effectiveHeight) {
        styles.aspectRatio = `${effectiveWidth} / ${effectiveHeight}`;
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
      const baseStyle = {
        left: `${this.popupX}px`,
        top: `${this.popupY}px`
      };
      
      // Adjust transform based on position
      switch (this.popupPosition) {
        case 'top':
          baseStyle.transform = 'translate(-50%, calc(-100% - 16px))';
          break;
        case 'bottom':
          baseStyle.transform = 'translate(-50%, 16px)';
          break;
        case 'right':
          baseStyle.transform = 'translate(16px, -50%)';
          break;
        case 'left':
          baseStyle.transform = 'translate(calc(-100% - 16px), -50%)';
          break;
      }
      
      return baseStyle;
    },
  },
  methods: {
    handleImageLoad(event) {
      this.naturalWidth = event.target.naturalWidth;
      this.naturalHeight = event.target.naturalHeight;
    },
    handleImageClick(event) {
      const wrapper = this.$refs.wrapper;
      if (!wrapper) {
        console.warn('[Heatmap] Wrapper not found', { maskUrl: this.maskUrl, backgroundUrl: this.backgroundUrl });
        return;
      }

      // Get image element first to calculate position relative to actual image
      const imgEl = wrapper.querySelector('.overlay-image');
      if (!imgEl) {
        console.warn('[Heatmap] Overlay image element not found', { maskUrl: this.maskUrl, backgroundUrl: this.backgroundUrl });
        return;
      }
      
      const imgRect = imgEl.getBoundingClientRect();
      
      // Get click position relative to actual image (not wrapper!)
      const relativeX = event.clientX - imgRect.left;
      const relativeY = event.clientY - imgRect.top;
      
      // Check if click is within image bounds with small tolerance for edge cases
      const tolerance = 1; // 1px tolerance for edge clicks with browser zoom
      if (relativeX < -tolerance || relativeY < -tolerance || 
          relativeX > imgRect.width + tolerance || relativeY > imgRect.height + tolerance) {
        return;
      }
      
      // Clamp coordinates to image bounds (handle edge cases from browser zoom)
      const clampedX = Math.max(0, Math.min(relativeX, imgRect.width - 0.01));
      const clampedY = Math.max(0, Math.min(relativeY, imgRect.height - 0.01));
      
      // Set visual indicator position (relative to image, not wrapper)
      this.clickX = clampedX;
      this.clickY = clampedY;
      this.popupX = clampedX;
      this.popupY = clampedY;
      
      // Determine best popup position based on click location
      const popupHeight = 40; // Approximate popup height
      const popupWidth = 80; // Approximate popup width (half width for centered popup)
      const margin = 20; // Minimum margin from edges
      
      // Check available space in each direction (relative to image, not wrapper)
      const spaceTop = clampedY;
      const spaceBottom = imgRect.height - clampedY;
      const spaceLeft = clampedX;
      const spaceRight = imgRect.width - clampedX;
      
      // Check if popup would overflow horizontally when positioned top/bottom
      const wouldOverflowLeft = clampedX < (popupWidth / 2);
      const wouldOverflowRight = (imgRect.width - clampedX) < (popupWidth / 2);
      
      // Logic: prefer top, but if at edges use left/right
      if (spaceTop > popupHeight + margin && !wouldOverflowLeft && !wouldOverflowRight) {
        // Enough space on top and won't overflow horizontally
        this.popupPosition = 'top';
      } 
      else if (wouldOverflowRight && spaceLeft > popupWidth + margin) {
        // Point is at right edge, show popup on left
        this.popupPosition = 'left';
      } 
      else if (wouldOverflowLeft && spaceRight > popupWidth + margin) {
        // Point is at left edge, show popup on right
        this.popupPosition = 'right';
      } 
      else if (spaceTop > popupHeight + margin) {
        // Use top even if might slightly overflow (better than nothing)
        this.popupPosition = 'top';
      } 
      else if (spaceBottom > popupHeight + margin && !wouldOverflowLeft && !wouldOverflowRight) {
        // If no space on top, show popup below (if won't overflow)
        this.popupPosition = 'bottom';
      } 
      else if (spaceRight > popupWidth + margin) {
        // Show on right if there's space
        this.popupPosition = 'right';
      } 
      else if (spaceLeft > popupWidth + margin) {
        // Show on left if there's space
        this.popupPosition = 'left';
      } 
      else if (spaceBottom > popupHeight + margin) {
        // Fallback to bottom even if might overflow
        this.popupPosition = 'bottom';
      } 
      else {
        // Final fallback: top
        this.popupPosition = 'top';
      }

      // Use mask dimensions from server
      const maskWidth = this.maskWidth;
      const maskHeight = this.maskHeight;
      
      if (!maskWidth || !maskHeight) {
        console.warn('[Heatmap] Mask dimensions not available', { 
          maskUrl: this.maskUrl,
          maskWidth: this.maskWidth,
          maskHeight: this.maskHeight
        });
        return;
      }
      
      // Get PNG file dimensions (naturalWidth/Height of the loaded image)
      const pngWidth = imgEl.naturalWidth;
      const pngHeight = imgEl.naturalHeight;
      
      if (!pngWidth || !pngHeight) {
        console.warn('[Heatmap] PNG dimensions not available', {
          maskUrl: this.maskUrl,
          naturalWidth: imgEl.naturalWidth,
          naturalHeight: imgEl.naturalHeight
        });
        return;
      }
      
      // Two-step coordinate transformation:
      // 1. From screen coordinates to PNG coordinates
      // 2. From PNG coordinates to mask coordinates
      
      // Step 1: Scale from displayed size to PNG size
      const displayToImageScaleX = pngWidth / imgRect.width;
      const displayToImageScaleY = pngHeight / imgRect.height;
      
      const pngX = clampedX * displayToImageScaleX;
      const pngY = clampedY * displayToImageScaleY;
      
      // Step 2: Scale from PNG size to mask size
      const imageTomaskScaleX = maskWidth / pngWidth;
      const imageTomaskScaleY = maskHeight / pngHeight;
      
      let maskX = Math.floor(pngX * imageTomaskScaleX);
      let maskY = Math.floor(pngY * imageTomaskScaleY);
      
      // Clamp to mask bounds
      maskX = Math.min(Math.max(maskX, 0), maskWidth - 1);
      maskY = Math.min(Math.max(maskY, 0), maskHeight - 1);

      // Update state - this will trigger server-side callback
      this.$emit('update:mask-x', maskX);
      this.$emit('update:mask-y', maskY);
      
      // Reset hiding state for new click
      this.isHiding = false;
      
      // Don't set clicked-value here - server will set it after getting value from mask
      this.$emit('update:clicked-value', null);

      // Call server callback after Vue updates state
      if (this.onImageClick) {
        this.$nextTick(() => {
          this.onImageClick();
        });
      }
    },
    handleMouseLeave() {
      if (this.clickedValue === null) return;
      
      this.isHiding = true;
      
      setTimeout(() => {
        this.$emit('update:clicked-value', null);
        this.$emit('update:mask-x', null);
        this.$emit('update:mask-y', null);
        this.isHiding = false;
      }, 300);
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
    opacity: Number,
    width: [Number, String],
    height: [Number, String],
    maskWidth: Number,
    maskHeight: Number,
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
