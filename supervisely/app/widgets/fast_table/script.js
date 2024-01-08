Vue.component('fast-table', {
  props: {
    data: {
      type: Array,
      default: () => [],
    },
    sort: {
      type: Object,
      default: () => ({}),
    },
    total: {
      type: Number,
      default: 1,
    },
    page: {
      type: Number,
      default: 1,
    },
    pageSize: {
      type: Number,
      default: 50,
    },
    options: {
      type: Object,
      default: () => ({}),
    },
    columnsOptions: {
      type: Array,
      default: () => [],
    },
    columns: {
      type: Array,
      default: () => [],
    },
    projectMeta: {
      type: Object,
      default: () => ({}),
    },
    search: {
      type: String,
      default: '',
    },
  },
  data() {
    return {
      columnNumberLimit: 15,
    };
  },
  computed: {
    canGoBack() {
      return this.page - 1 > 0;
    },
    canGoForward() {
      return this.page * this.LIMIT < this.total;
    },
    settings() {
      return {
        ...(this.options || {}),
      };
    },
    LIMIT() {
      return this.pageSize || 50;
    },
    classesMap() {
      return this.projectMeta ? _.keyBy(this.projectMeta.classes, 'title') : {};
    },
    needsRightGradient() {
      return false;
    },
    columnsSettings() {
      return this.columnsOptions || Array(this.columns.length || 0).fill({});
    },
    oneOfRowsHasSubtitle() {
      return this.columnsSettings.find(i => i?.subtitle);
    },
  },
  watch: {
    // scrollX() {
    //   if (this.scrollX.value > 0 && columnNumberLimit.value !== 99999) columnNumberLimit.value = 99999;
    // },
  },
  mounted() {
  },
  created() {
    this.updateData = _.throttle(this._updateData, 200);
  },
  methods: {
    _updateData() {
      this.$emit('filters-changed');
    },
    searchChanged(val) {
      this.$emit('update:search', val);
      this.updateData();
    },
    go(dir) {
      if (dir === -1 && !this.canGoBack) return;
      if (dir === +1 && !this.canGoForward) return;

      this.$emit('update:page', this.page + dir);
      this.updateData();
    },
    onHeaderCellClick(idx) {
      let order = this.sort.order;

      if (this.sort.column !== idx) order = 'asc';
      else order = this.sort.order === 'asc' ? 'desc' : 'asc';

      this.$emit('update:sort', { order, column: idx });
      this.updateData();
    },
    highlight(text) {
      if (!this.search) return text;
      return text.toString().replace(new RegExp(this.search, 'gi'), match => `<span class="bg-yellow-400">${match}</span>`);
    },
  },
  template: `
 <div class="tailwind fast-table">
    <div class="rounded-lg border border-slate-200 shadow bg-white" ref="wrapper">
      <div class="py-2 px-2 md:px-5 md:py-4 flex flex-col md:flex-row gap-2 justify-between items-center">
        <div class="relative w-full md:max-w-[18rem]">
          <i class="zmdi zmdi-search h-4 absolute top-2 left-2.5 opacity-50"></i>
          <!--<img alt="Search" src="~assets/icons/search.png" class="h-4 absolute top-2 left-2.5 opacity-50" />
          <img alt="Close" src="~assets/icons/close-light.svg" class="h-4 absolute top-2.5 right-3 opacity-50 cursor-pointer" v-if="search" @click="searchChanged('')" />
          -->
          <i class="zmdi zmdi-close h-4 absolute top-2.5 right-3 opacity-50 cursor-pointer" v-if="search" @click="searchChanged('')"></i>
          <input :value="search" @input="searchChanged($event.target.value)" type="text" placeholder="Search" class="text-sm rounded-md px-3 py-1.5 text-gray-900 shadow-sm placeholder:text-gray-400 border border-slate-200 w-full pl-8 bg-slate-50" @keydown.esc="searchChanged('')" />
        </div>
        <div class="text-[.9rem] text-slate-500 flex items-center gap-0 w-full md:w-auto whitespace-nowrap" v-if="(data?.length)">
          <button aria-label="Go back" @click="go(-1)" class="hover:text-secondary-500 text-[1.3rem] md:ml-0 -ml-2" :class="{ '!text-slate-200 cursor-default': !canGoBack }"><i class="zmdi zmdi-chevron-left" /></button>
          <span>Rows {{ Math.min((page - 1) * LIMIT + 1, total)  }}-{{ Math.min(total, page * LIMIT) }} of {{ total }}</span>
          <button aria-label="Go forward" @click="go(+1)" class="hover:text-secondary-500 text-[1.3rem]" :class="{ '!text-slate-200 cursor-default': !canGoForward }"><i class="zmdi zmdi-chevron-right" /></button>
          <div class="md:hidden w-full flex-1"></div>
        </div>
      </div>
      <div class="overflow-x-auto border-t border-slate-200 relative" ref="scrollBox">
        <div :class="{ 'opacity-100 !flex': needsRightGradient }" class="bg-gradient-to-r from-transparent to-white absolute right-0 top-0 bottom-0 z-10 w-56 opacity-0 transition-opacity hidden">
          <i class="zmdi zmdi-chevron-right absolute top-1/2 right-2 text-[1.5rem] text-slate-400 -translate-y-1/2" />
        </div>
        <table class="w-full text-[.8rem] md:text-[.9rem] mb-1" ref="tableBox">
          <thead>
            <tr>
              <th
                v-for="(c,idx) in columns.slice(0, columnNumberLimit)"
                class="px-2 md:px-3 py-2.5 whitespace-nowrap first:pl-3 last:pr-3 md:first:pl-6 md:last:pr-6 first:text-left cursor-pointer sticky top-0 bg-slate-50 box-content shadow-[inset_0_-2px_0_#dfe6ec] group"
                :class="{ 'first:sticky first:left-0 first:z-20 first:shadow-[inset_-2px_-2px_0_#dfe6ec]': settings.fixColumns }"
                @click="onHeaderCellClick(idx)"
              >
                <div class="flex items-center">
                  <span>{{ c }}</span>
                  <el-tooltip v-if="columnsSettings[idx]?.tooltip">
                    <div slot="content" v-html="columnsSettings[idx].tooltip" />
                    <i class="zmdi zmdi-info-outline ml5 text-slate-500 text-[12px]" />
                  </el-tooltip>
                  <span class="w-[16px] text-[12px] text-secondary-500 ml-1">
                    <i class="zmdi zmdi-sort-amount-asc ml5 opacity-0 group-hover:opacity-100 text-slate-400 transition-opacity" v-if="sort.column !== idx" />
                    <i class="zmdi zmdi-sort-amount-asc ml5" v-if="sort.column === idx && sort.order === 'asc'" />
                    <i class="zmdi zmdi-sort-amount-desc ml5" v-if="sort.column === idx && sort.order === 'desc'" />
                  </span>
                </div>
                <div class="text-[.7rem] text-slate-500 font-normal text-left" v-if="oneOfRowsHasSubtitle">
                  {{ columnsSettings[idx]?.subtitle || 'ㅤ' }}
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in data" class="border-b border-gray-200 last:border-0 group" :class="{ 'cursor-pointer': settings.isRowClickable }" @click="settings.isRowClickable && $emit('row-click', { idx: row.idx, row: row.items })">
              <td                
                v-for="(col,idx) in row.items.slice(0, columnNumberLimit)"
                class="px-2 md:px-3 py-2 bg-white first:pl-3 last:pr-3 md:first:pl-6 md:last:pr-6 group-hover:bg-slate-50 group"
                :class="{ 'first:sticky first:left-0 first:z-10 first:shadow-[inset_-2px_0_0_#dfe6ec]': settings.fixColumns, 'cursor-pointer': settings.isCellClickable }"
                @click="settings.isCellClickable && $emit('cell-click', { idx: row.idx, row: row.items , column: idx})"
              >
                <div class="flex items-center whitespace-nowrap">
                  <span v-if="columnsSettings[idx]?.type === 'class'" class="w-3 h-3 rounded-sm flex mr-1.5 flex-none" :style="{ backgroundColor: (classesMap[col] || { color: '#00ff00' }).color }"></span>
                  <span v-html="highlight(col)"></span>
                  <span v-if="idx === 0 && settings.isRowClickable" class="opacity-0 transition-all duration-300 text-secondary-500 text-xs group-hover:translate-x-2 group-hover:opacity-100">➔</span>
                  <span class="text-slate-400 ml-0.5 text-[.7rem]" v-if="columnsSettings[idx]?.postfix">{{ columnsSettings[idx].postfix }}</span>
                </div>
                <div v-if="columnsSettings[idx]?.type === 'class'" class="text-[.6rem] text-slate-500 pl-[1.2rem]">
                  {{ (classesMap[col]?.shape || 'Unknown').replace('bitmap', 'mask') }}
                </div>
                <div v-if="columnsSettings[idx]?.maxValue" class="h-[2px] bg-slate-200 mt-0.5 w-[50px]">
                  <div class="h-[2px] bg-secondary-500" :style="{ width: col * 100 / columnsSettings[idx].maxValue + '%' }" />
                </div>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-if="!(data?.length)" class="text-[.9rem] text-center text-slate-500 mt-5 mb-5">
          😭 Nothing is found
        </div>
      </div>
    </div>
  </div> 
  `,
});