Vue.component("sly-card2", {
  props: {
    title: {
      type: String,
      default: "",
    },
    description: {
      type: String,
      default: "",
    },
    properties: {
      type: Object,
      default: () => ({}),
    },
    propertiesLayout: {
      type: String,
      default: "vertical", // 'vertical' or 'horizontal'
      validator: (value) => ["vertical", "horizontal"].includes(value),
    },
    collapsable: {
      type: Boolean,
      default: false,
    },
    removePadding: {
      type: Boolean,
      default: false,
    },
    overflow: {
      type: String,
      default: "auto",
    },
    lockMessage: {
      type: String,
      default: "Card content is locked",
    },
    isLocked: {
      type: Boolean,
      default: false,
    },
    hasIcon: {
      type: Boolean,
      default: false,
    },
    hasContent: {
      type: Boolean,
      default: false,
    },
    hasContentTopRight: {
      type: Boolean,
      default: false,
    },
  },
  data() {
    return {
      collapsed: false,
    };
  },
  computed: {
    hasProperties() {
      return this.properties && this.properties.length > 0;
    },
    hasHeader() {
      return (
        this.title ||
        this.description ||
        this.hasIcon ||
        this.hasContentTopRight ||
        this.collapsable
      );
    },
  },
  methods: {
    toggleCollapse() {
      this.collapsed = !this.collapsed;
    },
  },
  template: `
    <div class="sly-card2" :class="{'no-padding': removePadding}">
      <!-- Header -->
      <div class="sly-card2-header" v-if="hasHeader">
        <!-- Icon slot -->
        <div v-if="hasIcon" class="sly-card2-header-icon">
          <slot name="icon"></slot>
        </div>

        <div class="sly-card2-header-content">
          <h3 v-if="title" class="sly-card2-title">{{ title }}</h3>
          <div v-if="description" class="sly-card2-description">{{ description }}</div>
        </div>

        <!-- Top right content slot -->
        <div v-if="hasContentTopRight" class="sly-card2-header-slot">
          <slot name="content-top-right"></slot>
        </div>

        <!-- Collapse button -->
        <div v-if="collapsable" class="sly-card2-header-actions">
          <span @click="toggleCollapse">
            <i :class="collapsed ? 'zmdi zmdi-chevron-down' : 'zmdi zmdi-chevron-up'"></i>
          </span>
        </div>
      </div>

    <div :class="[collapsed ? 'collapsed-content' : '']" v-if="hasContent || hasProperties">
      <!-- Divider -->
      <div class="sly-card2-divider" v-if="hasProperties && hasHeader"></div>

      <template v-if="hasProperties" >
        <div class="sly-card2-properties" :class="[
          propertiesLayout === 'vertical' ? 'sly-card2-properties-vertical' : 'sly-card2-properties-horizontal'
        ]">
          <!-- For vertical layout -->
          <template v-if="propertiesLayout === 'vertical'">
            <div class="sly-card2-property-item" v-for="(prop, index) in properties" :key="prop.key">
              <span class="sly-card2-property-name" :key="'name-'+index">{{ prop.key }}:</span>
              <div class="sly-card2-property-value">
                <span :key="'value-'+index"> {{ prop.value }}</span>
                <span v-if="prop.extra" :key="'extra-'+index" class="sly-card2-property-value-tag">{{ prop.extra }}</span>
              </div>

            </div>
          </template>

          <!-- For horizontal layout -->
          <template v-else>
            <div class="sly-card2-properties-content">
              <template v-for="(prop, index) in properties">
                <!-- Separator -->
                <span v-if="index > 0" class="sly-card2-properties-separator" :key="'sep-'+index">•</span>
                <!-- Display each value in the array -->
                <span :key="'name-'+index" >{{ prop.value }}</span>
                <span :key="'value-'+index" >{{ prop.value }}</span>
                <span :key="'extra-'+index" v-if="prop.extra" :class=" sly-card2-property-value-tag">{{ prop.extra }}</span>
              </template>
            </div>
          </template>
        </div>
        </template>
        
        <!-- Content -->
        <div class="sly-card2-divider" v-if="hasContent && (hasProperties || hasHeader)"></div>
        <div class="sly-card2-content" v-if="hasContent"  
            
            :style="{ overflow: overflow }">
            <slot></slot>
        </div>
      </div>
    </div>
  `,
});
