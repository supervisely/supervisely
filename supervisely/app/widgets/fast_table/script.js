Vue.component('fast-table', {
  template: `
    <!-- eslint-disable vue/no-v-html -->
    <div class="sly-ninja-table" :disabled="disabled" style="position: relative;">
    <div v-if="disabled" class="sly-fast-table-disable-overlay"></div>
      <div class=" tailwind fast-table">
        <div
          ref="wrapper"
          class="rounded-lg border border-slate-200 bg-white overflow-hidden"
        >
          <div
            v-if="settings.showHeaderControls"
            class="py-2 px-2 md:px-5 md:py-4 flex flex-col md:flex-row gap-2 justify-between items-center"
          >
            <div
              class="fflex"
              style="flex-grow: 1"
            >
              <slot name="header-left-side-start" />
              <div v-if="settings.searchPosition !== 'right'" class="relative w-full md:max-w-[14rem]">
                <i class="zmdi zmdi-search h-4 absolute top-2 left-2.5 opacity-50" />
                <i
                  v-if="search"
                  class="zmdi zmdi-close h-4 absolute top-2.5 right-3 opacity-50 cursor-pointer"
                  @click="searchChanged('')"
                />
                <input
                  :value="search"
                  type="text"
                  :placeholder="\`\${name ? \`Search for \${name}...\` : 'Search'}\`"
                  class="text-sm rounded-md px-3 py-1.5 text-gray-900 shadow-sm placeholder:text-gray-400 border border-slate-200 w-full pl-8 bg-slate-50"
                  @input="searchChanged($event.target.value)"
                  @keydown.esc="searchChanged('')"
                >
              </div>
              <slot name="header-left-side-end" />
            </div>
            <div v-if="settings.searchPosition === 'right'" class="relative w-full md:max-w-[14rem]">
              <i class="zmdi zmdi-search h-4 absolute top-2 left-2.5 opacity-50" />
              <i
                v-if="search"
                class="zmdi zmdi-close h-4 absolute top-2.5 right-3 opacity-50 cursor-pointer"
                @click="searchChanged('')"
              />
              <input
                :value="search"
                type="text"
                :placeholder="\`\${name ? \`Search for \${name}...\` : 'Search'}\`"
                class="text-sm rounded-md px-3 py-1.5 text-gray-900 shadow-sm placeholder:text-gray-400 border border-slate-200 w-full pl-8 bg-slate-50"
                @input="searchChanged($event.target.value)"
                @keydown.esc="searchChanged('')"
              >
            </div>
            <div
              v-if="data && data.length"
              class="text-[.9rem] text-slate-500 flex items-center gap-0 w-full md:w-auto whitespace-nowrap"
            >
              <button
                aria-label="Go back"
                class="hover:text-secondary-500 text-[1.3rem] md:ml-0 -ml-2"
                :class="{ '!text-slate-200 cursor-default': !canGoBack }"
                @click="go(-1)"
              >
                <i class="zmdi zmdi-chevron-left" />
              </button>
              <span>Rows {{ Math.min((page - 1) * LIMIT + 1, total) }}-{{ Math.min(total, page * LIMIT) }} of {{ total }}</span>
              <button
                aria-label="Go forward"
                class="hover:text-secondary-500 text-[1.3rem]"
                :class="{ '!text-slate-200 cursor-default': !canGoForward }"
                @click="go(+1)"
              >
                <i class="zmdi zmdi-chevron-right" />
              </button>
              <div class="md:hidden w-full flex-1" />
            </div>
          </div>
          <div
            ref="scrollBox"
            class="overflow-x-auto border-slate-200 relative"
            :class="{ 'border-t': settings.showHeaderControls}"
          >
            <div
              :class="{ 'opacity-100 !flex': needsRightGradient }"
              class="bg-gradient-to-r from-transparent to-white absolute right-0 top-0 bottom-0 z-10 w-56 opacity-0 transition-opacity hidden"
            >
              <i class="zmdi zmdi-chevron-right absolute top-1/2 right-2 text-[1.5rem] text-slate-400 -translate-y-1/2" />
            </div>
            <table
              ref="tableBox"
              class="w-full text-[.8rem] md:text-[.9rem] mb-1"
              :key="'table-' + page"
            >
              <thead>
                <tr>
                  <th
                    v-if="settings.isRadio"
                    class="px-2 md:px-3 py-2.5 whitespace-nowrap first:pl-3 last:pr-3 md:last:pr-6 first:text-left cursor-pointer sticky top-0 bg-slate-50 box-content shadow-[inset_0_-2px_0_#dfe6ec] group"
                    :class="{ 'first:sticky first:left-0 first:z-20 first:shadow-[inset_-2px_-2px_0_#dfe6ec]': fixColumns }"
                    style="width: 20px;"
                  >
                  </th>
                  <th
                    v-if="settings.isRowSelectable && (!settings.maxSelectedRows || settings.maxSelectedRows > 1)"
                    class="px-2 md:px-3 py-2.5 whitespace-nowrap first:pl-3 last:pr-3 md:last:pr-6 first:text-left cursor-pointer sticky top-0 bg-slate-50 box-content shadow-[inset_0_-2px_0_#dfe6ec] group"
                    :class="{ 'first:sticky first:left-0 first:z-20 first:shadow-[inset_-2px_-2px_0_#dfe6ec]': fixColumns }"
                  >
                    <el-checkbox
                      v-if="settings.isRowSelectable && (!settings.maxSelectedRows || settings.maxSelectedRows > 1)"
                      size="small"
                      v-model="headerCheckboxModel"
                      :indeterminate="isSomeOnPageSelected && !isAllOnPageSelected"
                      :key="'headercb-' + page + '-' + (data && data.map(r => r.idx).join(',')) + '-' + (selectedRows && selectedRows.map(r => r.idx).join(',')) + '-' + (isAllOnPageSelected?1:0) + '-' + (isSomeOnPageSelected?1:0)"
                      style="transform: scale(1.2);"
                    />
                  </th>
                  <th
                    v-if="settings.isRowSelectable && settings.maxSelectedRows === 1"
                    class="px-2 md:px-3 py-2.5 whitespace-nowrap first:pl-3 last:pr-3 md:last:pr-6 first:text-left cursor-pointer sticky top-0 bg-slate-50 box-content shadow-[inset_0_-2px_0_#dfe6ec] group"
                    :class="{ 'first:sticky first:left-0 first:z-20 first:shadow-[inset_-2px_-2px_0_#dfe6ec]': fixColumns }"
                  >
                  </th>
                  <th
                    v-for="(c,idx) in columns.slice(0, columnNumberLimit)"
                    class="px-2 md:px-3 py-2.5 whitespace-nowrap first:pl-3 last:pr-3 md:first:pl-6 md:last:pr-6 first:text-left cursor-pointer sticky top-0 bg-slate-50 box-content shadow-[inset_0_-2px_0_#dfe6ec] group"
                    :class="{ 'first:sticky first:left-0 first:z-20 first:shadow-[inset_-2px_-2px_0_#dfe6ec]': fixColumns }"
                    @click="onHeaderCellClick(idx)"
                  >
                    <div class="flex items-center">
                      <span>{{ c }}</span>
                      <el-tooltip v-if="columnsSettings[idx] && columnsSettings[idx].tooltip">
                        <div
                          slot="content"
                          v-html="columnsSettings[idx].tooltip"
                        />
                        <i
                          class="zmdi zmdi-info-outline ml5 text-slate-500 text-[12px]"
                          style="margin-top: -1px;"
                        />
                      </el-tooltip>
                      <span
                        v-if="!columnsSettings[idx] || !columnsSettings[idx].disableSort"
                        class="w-[16px] text-[12px] text-secondary-500 ml-1"
                      >
                        <i
                          v-if="sort.column !== idx"
                          class="zmdi zmdi-sort-amount-asc ml5 opacity-0 group-hover:opacity-100 text-slate-400 transition-opacity"
                        />
                        <i
                          v-if="sort.column === idx && sort.order === 'asc'"
                          class="zmdi zmdi-sort-amount-asc ml5"
                        />
                        <i
                          v-if="sort.column === idx && sort.order === 'desc'"
                          class="zmdi zmdi-sort-amount-desc ml5"
                        />
                      </span>
                    </div>
                    <div
                      v-if="oneOfRowsHasSubtitle"
                      class="text-[.7rem] text-slate-500 font-normal text-left"
                    >
                      {{ columnsSettings[idx] && columnsSettings[idx].subtitle ? columnsSettings[idx].subtitle : 'ㅤ' }}
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody :key="'page-' + page">
                <tr
                  v-for="row in data"
                  :key="'row-' + rowKeyValue(row)"
                  class="border-b border-gray-200 last:border-0 group"
                  :class="{ 'cursor-pointer': settings.isRowClickable }"
                  @click="settings.isRowClickable && $emit('row-click', { idx: rowKeyValue(row), row: row.items, columnsSettings })"
                >
                  <td
                    v-if="settings.isRadio"
                    class="px-2 md:px-3 py-2 bg-white first:pl-3 last:pr-3 md:last:pr-6 first:text-left cursor-pointer group-hover:bg-slate-50 group"
                    :class="{ 'first:sticky first:left-0 first:z-10 first:shadow-[inset_-2px_0_0_#dfe6ec]': fixColumns, 'cursor-pointer': settings.isCellClickable }"
                  >
                    <el-radio
                      v-model="selectedRadioIdx"
                      :label="row.idx"
                      @input="updateSelectedRadio(row)"
                      style="width: 20px;"
                      class="row-radio"
                    >
                      <span style="margin-left: -20px; visibility: hidden;">{{ row.idx }}</span>
                    </el-radio>
                  </td>
                  <td
                    v-else-if="settings.isRowSelectable"
                    class="row-select-cell px-2 md:px-3 py-2 bg-white first:pl-3 last:pr-3 md:last:pr-6 group-hover:bg-slate-50 group"
                    :class="{ 'first:sticky first:left-0 first:z-10 first:shadow-[inset_-2px_0_0_#dfe6ec]': fixColumns, 'cursor-pointer': settings.isCellClickable }"
                  >
                    <el-checkbox
                      size="small"
                      v-model="rowCheckboxModel[rowKeyValue(row)]"
                      @change="onRowCheckboxChange(row, rowCheckboxModel[rowKeyValue(row)])"
                      @click.stop
                      :key="'rowcb-' + page + '-' + rowKeyValue(row)"
                      style="transform: scale(1.2);"
                    />
                  </td>

                  <td
                    v-for="(col,idx) in row.items.slice(0, columnNumberLimit)"
                    class="px-2 md:px-3 py-2 bg-white first:pl-3 last:pr-3 md:first:pl-6 md:last:pr-6 group-hover:bg-slate-50 group"
                    :class="{ 'first:sticky first:left-0 first:z-10 first:shadow-[inset_-2px_0_0_#dfe6ec]': fixColumns, 'cursor-pointer': settings.isCellClickable }"
                    @click="settings.isCellClickable && $emit('cell-click', { idx: row.idx, row: row.items , column: idx})"
                  >
                    <div
                      class="flex items-center whitespace-nowrap"
                      :class="{ 'justify-end': columnsSettings[idx] && columnsSettings[idx].align === 'right' }"
                    >
                      <span
                        v-if="columnsSettings[idx] && columnsSettings[idx].type === 'class'"
                        class="w-3 h-3 rounded-sm flex mr-1.5 flex-none"
                        :style="{ backgroundColor: (classesMap[col] || { color: '#00ff00' }).color }"
                      />
                      <span v-if="columnsSettings[idx] && columnsSettings[idx].customCell">
                        <slot
                          name="cell-content"
                          :row="row"
                          :cell-value="col"
                          :column="columns[idx]"
                          :idx="idx"
                        />
                      </span>
                      <el-tooltip
                        v-else-if="columnsSettings[idx] && columnsSettings[idx].maxWidth"
                        :enterable="false"
                        :open-delay="300"
                      >
                        <span
                          slot="content"
                          v-html="col"
                        />
                        <span
                          class="ellipsis"
                          :style="{ 'max-width': columnsSettings[idx].maxWidth }"
                          v-html="highlight(col)"
                        />
                      </el-tooltip>
                      <span
                        v-else
                        v-html="highlight(col)"
                      />
                      <span
                        v-if="col != '' && columnsSettings[idx] && columnsSettings[idx].postfix"
                        class="text-slate-400 ml-0.5 text-[.7rem]"
                      >{{ columnsSettings[idx].postfix }}</span>
                      <span
                        v-if="idx === 0 && settings.isRowClickable"
                        class="opacity-0 transition-all duration-300 text-secondary-500 text-xs group-hover:translate-x-2 group-hover:opacity-100"
                      >➔</span>
                    </div>
                    <div
                      v-if="columnsSettings[idx] && columnsSettings[idx].type === 'class'"
                      class="text-[.6rem] text-slate-500 pl-[1.2rem]"
                    >
                      {{ (classesMap[col] && classesMap[col].shape ? classesMap[col].shape : 'Unknown').replace('bitmap', 'mask') }}
                    </div>
                    <div
                      v-if="columnsSettings[idx] && columnsSettings[idx].maxValue"
                      class="h-[2px] bg-slate-200 mt-0.5 w-[50px]"
                    >
                      <div
                        class="h-[2px] bg-secondary-500"
                        :style="{ width: col * 100 / columnsSettings[idx].maxValue + '%' }"
                      />
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
            <div
              v-if="!data || !data.length"
              class="text-[.9rem] text-center text-slate-500 mt-5 mb-5"
            >
              😭 Nothing is found
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  
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
    selectedRows: {
      type: Array,
      default: () => [],
    },
    selectedRadioIdx: {
      type: Number,
      default: 0,
    },
    name: {
      type: String,
      default: '',
    },
    disabled: {
      type: Boolean,
      default: false,
    },
    rowKey: {
      type: String,
      default: 'idx',
    },
  },

  data() {
    return {
      columnNumberLimit: 50,
      rowCheckboxModel: {},
    };
  },

  computed: {
    canGoBack() {
      return this.page - 1 > 0;
    },

    canGoForward() {
      return this.page * this.LIMIT < this.total;
    },

    fixColumns() {
      if (!this.settings?.fixColumns) return 0;

      if (this.settings?.isRowSelectable) return this.settings.fixColumns + 1;

      return this.settings.fixColumns;
    },

    settings() {
      const defaultSettings = {
        showHeaderControls: true,
        searchPosition: 'left',
      };
      return {
        ...defaultSettings,
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

    isAllOnPageSelected() {
      const rows = this.data || [];
      if (rows.length === 0) return false;
      const selectedIdx = new Set((this.selectedRows || []).map(r => this.rowKeyValue(r)));
      return rows.every(r => selectedIdx.has(this.rowKeyValue(r)));
    },

    isSomeOnPageSelected() {
      const rows = this.data || [];
      if (rows.length === 0) return false;
      const selectedIdx = new Set((this.selectedRows || []).map(r => this.rowKeyValue(r)));
      const count = rows.filter(r => selectedIdx.has(this.rowKeyValue(r))).length;
      return count > 0 && count < rows.length;
    },

    selectedIdxSet() {
      return new Set((this.selectedRows || []).map(r => this.rowKeyValue(r)));
    },

    headerCheckboxModel: {
      get() {
        return this.isAllOnPageSelected;
      },
      set(val) {
        this.selectAllRows(val);
      }
    },
  },

  watch: {
    data: {
      immediate: true,
      handler() {
        this.syncRowCheckboxModel();
      }
    },
    selectedRows: {
      immediate: true,
      handler() {
        this.syncRowCheckboxModel();
      }
    }
  },

  methods: {
    _updateData() {
      this.$emit('filters-changed')
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
      if (this.columnsSettings?.[idx]?.disableSort) return;

      let order = this.sort.order;

      if (this.sort.column !== idx) order = 'asc';
      else order = this.sort.order === 'asc' ? 'desc' : 'asc';

      this.$emit('update:sort', { order, column: idx });
      this.updateData();
    },

    selectAllRows(checked) {
      if (!this.data || this.data.length === 0) return;

      const current = Array.isArray(this.selectedRows) ? [...this.selectedRows] : [];
      const max = this.settings && this.settings.maxSelectedRows ? this.settings.maxSelectedRows : 0;

      if (checked) {
        const selectedIdx = new Set(current.map(r => this.rowKeyValue(r)));
        const result = [...current];

        if (max && max > 0) {
          let quota = Math.max(0, max - result.length);
          for (const r of this.data) {
            const key = this.rowKeyValue(r);
            if (!selectedIdx.has(key)) {
              if (quota <= 0) break;
              result.push(_.cloneDeep(r));
              selectedIdx.add(key);
              quota -= 1;
            }
          }
          this.$emit('update:selected-rows', result);
        } else {
          for (const r of this.data) {
            const key = this.rowKeyValue(r);
            if (!selectedIdx.has(key)) {
              result.push(_.cloneDeep(r));
              selectedIdx.add(key);
            }
          }
          this.$emit('update:selected-rows', result);
        }
      } else {
        const pageIdx = new Set(this.data.map(r => this.rowKeyValue(r)));
        const result = current.filter(r => !pageIdx.has(this.rowKeyValue(r)));
        this.$emit('update:selected-rows', result);
      }
    },

    syncRowCheckboxModel() {
      const map = {};
      const selected = new Set((this.selectedRows || []).map(r => this.rowKeyValue(r)));
      for (const r of (this.data || [])) {
        const key = this.rowKeyValue(r);
        map[key] = selected.has(key);
      }
      this.rowCheckboxModel = map;
    },

    onRowCheckboxChange(row, checked) {
      this.updateSelectedRows(row, checked);
    },

    updateSelectedRows(row, checked) {
      checked = typeof checked === 'boolean' ? checked : !!checked;
      const current = Array.isArray(this.selectedRows) ? [...this.selectedRows] : [];
      const key = this.rowKeyValue(row);
      const exists = current.some(r => this.rowKeyValue(r) === key);
      const max = this.settings && this.settings.maxSelectedRows ? this.settings.maxSelectedRows : 0;

      let result = current;
      if (checked) {
        if (max && max > 0) {
          if (max === 1) {
            result = [_.cloneDeep(row)];
            this.$emit('update:selected-rows', result);
            return;
          }
          if (!exists && current.length >= max) {
            // revert checkbox state if over the limit
            this.$nextTick(() => { this.$set(this.rowCheckboxModel, key, false); });
            return;
          }
        }
        if (!exists) result = [...current, _.cloneDeep(row)];
      } else {
        if (exists) result = current.filter(r => this.rowKeyValue(r) !== key);
      }
      this.$emit('update:selected-rows', result);
    },

    updateSelectedRadio(row) {
      console.log('updateSelectedRadio', row);
      row = _.cloneDeep(row)
      this.$emit('update:selected-rows', [row])
    },

    handleScroll() {
      const scrollX = this.$refs.scrollBox.scrollLeft;
      if (scrollX > 0 && this.columnNumberLimit !== 99999) {
        this.columnNumberLimit = 99999;
      }
    },

    highlight(text) {
      if (!this.search) return text;
      return text.toString().replace(new RegExp(this.search, 'gi'), match => '<span class="bg-yellow-400">'+match+'</span>');
    },

    rowKeyValue(row) {
      if (!row) return undefined;
      const keyName = this.rowKey || 'idx';
      return row[keyName] != null ? row[keyName] : row.idx;
    },
  },

  mounted() {
    this.$refs.scrollBox.addEventListener('scroll', this.handleScroll);
  },

  beforeDestroy() {
    this.$refs.scrollBox.addEventListener('scroll', this.handleScroll);
  },

  created() {
      this.updateData = _.debounce(this._updateData, 300);
  },
});