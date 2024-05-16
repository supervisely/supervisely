import * as jsonpatch from 'https://cdn.jsdelivr.net/npm/fast-json-patch@3.1.1/index.mjs';
import throttle from 'https://cdn.jsdelivr.net/npm/lodash-es@4.17.21/throttle.js';
import cloneDeep from 'https://cdn.jsdelivr.net/npm/lodash-es@4.17.21/cloneDeep.js';
import isEqual from 'https://cdn.jsdelivr.net/npm/lodash-es@4.17.21/isEqual.js';
import jwtDecode from 'https://cdn.jsdelivr.net/npm/jwt-decode@3.1.2/build/jwt-decode.esm.js';

const eventBus = new Vue();
const vuePatchOptsSet = new Set(['add', 'remove', 'replace', 'move', 'copy']);
const completedAppStatusSet = new Set(['error', 'finished', 'terminating', 'stopped']);

Object.defineProperties(Vue.prototype, {
  $eventBus: {
    value: eventBus,
  },
});

function connectToSocket(url, ...namespaces) {
  const socket = io(`${url}/${namespaces.join('-')}`, {
    path: '/api/ws',
  });

  socket.on('connect', () => {
    socket.emit('authenticate', { token: localStorage.token });
  }).on('unauthorized', () => {
    socket.close();
    setTimeout(() => { socket.open(); }, 5000);
  });

  return socket;
}

function formatError(res, data = {}) {
  const err = new Error();

  err.status = res.status;
  err.title = res.statusText;
  err.details = data.details || data.detail;

  if (err?.status === 404 && res?.url) {
    err.details = {
      message: `"${res.url}" not found`,
    };
  } else if (!err.details) {
    err.details = {
      message: 'Something went wrong',
      skipError: true,
    };
  } else if (typeof err.details !== 'object') {
    const errMsg = err.details;
    err.details = {
      message: errMsg,
    };
  }

  return err;
}

async function requestErrorHandler(res) {
  if (!res.ok) {
    let data;
    try {
      data = await res.json();
    } catch (err) {}

    throw formatError(res, data);
  }

  return res;
}

function applyPatch(document, patch) {
  let curDocument = document;

  patch.forEach((operation) => {
    if (vuePatchOptsSet.has(operation.op)) {
      const pathParts = operation.path.split('/');
      const propName = pathParts.splice(-1)[0];

      let parentObject;

      if (pathParts.length > 1) {
        parentObject = jsonpatch.getValueByPointer(curDocument, pathParts.join('/'));
      } else {
        parentObject = curDocument;
      }

      if (operation.op === 'add') {
        if (operation.path === '') {
          curDocument = operation.value;
        } else {
          if (Array.isArray(parentObject)) {
            if (propName === '-') {
              if (Array.isArray(operation.value)) {
                parentObject.push(...operation.value);
              } else {
                parentObject.push(operation.value);
              }
            } else {
              parentObject.splice(propName, 0, operation.value);
            }
          } else {
            Vue.set(parentObject, propName, operation.value);
          }
        }
      } else if (operation.op === 'replace') {
        if (operation.path === '') {
          curDocument = operation.value;
        } else {
          if (Array.isArray(parentObject)) {
            parentObject.splice(propName, 1, operation.value);
          } else {
            Vue.delete(parentObject, propName);
            Vue.set(parentObject, propName, operation.value);
          }
        }
      } else if (operation.op === 'move' || operation.op === 'copy') {
          const pathPartsFrom = operation.from.split('/');
          const propNameFrom = pathPartsFrom.splice(-1)[0];

          let parentObjectFrom;

          if (pathParts.length > 1) {
            parentObjectFrom = jsonpatch.getValueByPointer(curDocument, pathPartsFrom.join('/'));
          } else {
            parentObjectFrom = curDocument;
          }

          const moveValue = cloneDeep(jsonpatch.getValueByPointer(curDocument, operation.from));

          if (operation.op === 'move') {
            Vue.delete(parentObjectFrom, propNameFrom);
          }

          if (Array.isArray(parentObject)) {
            parentObject.splice(propName, 0, moveValue);
          } else {
            Vue.set(parentObject, propName, moveValue);
          }
      } else if (operation.op === 'remove') {
        Vue.delete(parentObject, propName);
      }
    } else {
      curDocument = jsonpatch.applyOperation(document, operation, false, false).newDocument;
    }
  });

  return curDocument;
}

Vue.component('sly-debug-panel-content', {
  props: ['value'],
  template: `
    <div>
      <div ref="jsoneditor" style="width: 340px; height: calc(100vh - 40px)"></div>
    </div>
  `,
  watch: {
    value: {
      deep: true,
      handler (value) {
        this.editor.set(value);
      },
    },
  },
  mounted() {
    const container = this.$refs.jsoneditor;

    const options = {
      mode: 'view'
    };

    this.editor = new JSONEditor(container, options);
    this.editor.set(this.value);
  }
})

Vue.component('sly-debug-panel', {
  props: ['value'],
  template: `
    <div style="z-index: 9999999; position: fixed; top: 0; right: 0; background: rgba(255,255,255,0.5); padding: 5px; border-radius: 4px;">
      <div style="display: flex; justify-content: flex-end;">
        <el-button type="text" @click="isOpen = !isOpen" style="padding: 0;">
          <i :class="[isOpen ? 'el-icon-caret-top' : 'el-icon-caret-bottom']"></i>
        </el-button>
      </div>

      <sly-debug-panel-content v-if="isOpen" :value="value"></sly-debug-panel-content>
    </div>
  `,
  data: function () {
    return {
      isOpen: false,
    };
  },
});

Vue.component('sly-html-compiler', {
  props: ['template', 'state', 'data', 'context', 'params'],
  data() {
    return {
      templateRenderer: null,
    };
  },
  computed: {
    isHtml() {
      return this.template && typeof this.template === 'string' && this.template.trim().startsWith('<');
    },
  },
  render(createElement) {
    if (!this.templateRenderer) return '';

    if(this.isHtml) {
      return this.templateRenderer.call(this, createElement);
    } else {
      return this._v(this.template);
    }
  },

  watch: {
    template: {
      handler() {
        if (!this.template) return;

        const compiled = Vue.compile(this.template);
        this.$options.staticRenderFns = compiled.staticRenderFns;

        this.templateRenderer = compiled.render;
      },
      immediate: true,
    },
  }
});

Vue.component('sly-app-error', {
  template: `
<div>
  <el-dialog v-if="elementAvailable" v-model="visible" @close="onClose" :title="errorTitle" size="tiny">
    <div class="fflex" style="margin: -20px 0 -20px 0;">
      <i class="notification-box-icon el-icon-information information mr15" style="font-size: 35px; color: #50bfff;"></i>

      <span style="min-height: 35px; display: flex; align-items: center; word-break: break-word;">
        {{ errorMessage }}
      </span>
    </div>

    <div slot="footer">
      <el-button type="primary" @click="close">Ok</el-button>
    </div>
  </el-dialog>

  <div v-else-if="visible" class="notification-box notification-box-warning" style="background-color:lightgray;padding:10px;border-radius:6px;display:flex;align-items:center;background-color:rgb(255 236 236);border:1px solid rgb(255 214 214);border-left:4px solid rgb(238, 131, 131);">
    <i class="notification-box-icon zmdi zmdi-alert-triangle" style="font-size: 25px;margin-right: 10px;color: rgb(238, 131, 131);"></i>

    <div>
      <div class="notification-box-title" style="font-size: 16px; font-weight: bold;">{{errorTitle}}</div>
      {{ errorMessage }}
    </div>
  </div>
</div>
  `,

  data() {
    return {
      visible: false,
      err: null,
    };
  },

  computed: {
    elementAvailable () {
      return !!window.sly;
    },

    errorTitle() {
      if (!this.err) return '';

      return this.err.details.title || this.err.title || '';
    },

    errorMessage() {
      if (!this.err) return '';

      return this.err.details.message;
    },
  },

  methods: {
    open(err) {
      if (!err?.details?.message) return;
      this.err = err;

      this.$nextTick(() => {
        this.visible = true;
      });
    },

    onClose() {
      this.err = null;
    },

    close() {
      this.visible = false;
      this.onClose();
    },
  },
});

Vue.component('sly-app', {
  props: {
    url: {
      type: String,
      default: document.location.href,
    },
    hotkeys: {
      type: Array,
      default: () => [],
    },
  },

  template: `
<div>
  <sly-app-error ref="err-dialog"></sly-app-error>
  <div ref="app-content">
    <slot v-if="!loading" :state="state" :data="data" :command="command" :post="post" :session="task" />
  </div>

  <sly-debug-panel v-if="isDebugMode" :value="{ state: state, data: data }" />
</div>
  `,

  data: function () {
    return {
      loading: true,
      task: null,
      state: {
        scrollIntoView: null,
        slyNotification: null,
        datasets: null,
      },
      data: {},
      sessionInfo: {},
      context: {},
      ws: null,
      isDebugMode: false,
      publicApiInstance: null,
      apiInstance: null,
      appUrl: '',
    };
  },

  computed: {
    formattedUrl () {
      let formattedUrl = '';

      if (this.appUrl) {
        formattedUrl = this.appUrl.replace(/\/$/, '');
      }

      Object.defineProperties(Vue.prototype, {
        $appUrl: {
          get() {
            return formattedUrl;
          },
        },
      });

      return formattedUrl;
    },
  },

  watch: {
    'task.status': {
      handler(newStatus) {
        const isCompleted = completedAppStatusSet.has(newStatus);

        if (!isCompleted) return;

        this.$nextTick(() => {
          setTimeout(() => {
            const appEl = this.$refs['app-content'];
            if (!appEl) return;

            const elements = appEl.querySelectorAll('.el-button,.el-input,.el-input__inner,.el-textarea,.el-textarea__inner,.el-input-number,.el-radio__input,.el-radio__original,.el-switch,.el-switch:not(.available-in-offline) .el-switch__input,.el-slider:not(.available-in-offline) .el-slider__runway,.el-checkbox__input,.el-checkbox__original');

            Array.prototype.slice.call(elements).forEach((el) => {
              if (el.classList.contains('available-in-offline')) return;

              el.setAttribute('disabled', true);
              el.classList.add('is-disabled');
              el.classList.add('disabled');
            });
          }, 1000);
        });
      },
      immediate: true,
    },
    'state.scrollIntoView': {
      handler() {
        this.$nextTick(() => {
          const ref = this.state?.scrollIntoView;
    
          if (!ref) return;
      
          const component = this.$refs['app-content'].querySelector(`#${ref}`);
      
          if (!component) return;
      
          component.scrollIntoView({ behavior: 'smooth', block: 'start' });

          this.state.scrollIntoView = null;
        });
      },
      immediate: true,
    },
    'state.slyNotification': {
      handler() {
        this.$nextTick(() => {
          if (!this.state.slyNotification) return;

          this.$message(this.state.slyNotification);

          this.state.slyNotification = null;
        });
      },
      immediate: true,
      deep: true,
    },
  },

  methods: {
    checkPreviewMode() {
      if (!this.task || !completedAppStatusSet.has(this.task.status)) return false;

      this.$refs['err-dialog'].open({
        details: {
          message: 'Current application session is finished and available only in preview mode. You need to run this app again',
        },
      });

      return true;
    },

    async command(command, payload = {}) {
      console.log('Command!', command);

      if (this.checkPreviewMode()) return;

      this.$nextTick(() => {
        this.ws.send(JSON.stringify({
          command: command,
          state: this.state,
          context: this.context,
          payload,
        }));
      })
    },

    async post(command, payload = {}) {
      console.log('Http!', command);

      if (this.checkPreviewMode()) return;

      this.$nextTick(() => {
        fetch(`${this.formattedUrl}${command}`, {
            method: 'POST',
            body: JSON.stringify({
              state: this.state,
              context: this.context,
              payload,
            }),
            headers: {'Content-Type': 'application/json'}
        })
        .then(requestErrorHandler)
        .then(res => res.json())
        .then((json) => {
          if (!json) return;

          this.merge(json);
        })
        .catch((err) => {
          if (!err?.details?.skipError) {
            this.$refs['err-dialog'].open(err);
          }

          throw err;
        });
      })
    },

    async getJson(path, contentOnly = true) {
      if (this.checkPreviewMode()) return;

      return fetch(`${this.formattedUrl}${path}`, {
        method: 'POST',
      })
      .then(requestErrorHandler)
        .then(res => {
          if (contentOnly) {
            return res.json();
          }

          return res;
        })
        .then(res => res)
        .catch((err) => {
          if (!err?.details?.skipError) {
            this.$refs['err-dialog'].open(err);
          }

          console.error(err);
        });
    },

    async checkMerge(vuePatch, jsonPatch, key) {
      if (!isEqual(vuePatch, jsonPatch)) {
        const vueState = JSON.stringify(vuePatch);
        const jsonState = JSON.stringify(jsonPatch);

        console.log('merge diff:', { key, taskId: this.task?.id }, vueState, jsonState);

        try {
          await this.apiInstance.post(
            '/client-logs',
            [{
              level: 'warn',
              message: 'sly-app patch error',
              payload: { key, taskId: this.task?.id, vueState, jsonState },
              service: 'sly-app',
              timestamp: new Date(),
            }],
            {},
          ).then(r => r.data);
        } catch (err) {}
      }
    },

    shutdownApp() {
      this.post('/sly/shutdown');
    },

    runAction({ action, payload }) {
      if (action === 'shutdown') {
        this.shutdownApp();

        return;
      } else if (action) {
        this.$eventBus.$emit(action, payload);
      }
    },

    async merge(payload) {
      if (payload.state) {
        this.state = applyPatch(this.state, payload.state);
      }

      if (payload.data) {
        this.data = applyPatch(this.data, payload.data);
      }
    },

    updateTaskData(payload) {
      const taskId = parseInt(payload?.id || payload?.[0]?.id, 10);
      const taskStatus = payload?.status || payload?.[0]?.status;
      if (!this.task?.id || this.task?.id !== taskId || !taskStatus) return;

      console.log('Task WS update status:', taskStatus);
      this.task.status = taskStatus;
    },

    connectToWs() {
      this.ws = new WebSocket(`ws${document.location.protocol === "https:" ? "s" : ""}://${this.appUrl.replace("http://", "").replace("https://", "").replace(/\/$/, '')}/sly/ws`);

      this.ws.onmessage = (event) => {
        console.log('Message received from Python', event);

        if (!event.data || typeof event.data !== 'string') return;

        let parsedData;
        try {
          parsedData = JSON.parse(event.data);
        } catch (err) {
          console.error(err);
          return;
        }

        if (parsedData.runAction) {
          this.runAction(parsedData.runAction);
          return;
        }

        this.merge(parsedData);
      };

      this.ws.onopen = () => {
        clearInterval(this.wsTimerId);

        if (!this.isDebugMode) {
          this.ws.onclose = () => {
            console.log('WS connection closed');

            this.wsTimerId = setInterval(() => {
              this.connectToWs();
            }, 8000);
          };
        }
      };
    }, 

    hotkeysHandler(e) {
      const k = this.hotkeys.filter(h => h.keyCode === e.keyCode);
      let hotkey;
      
      for(let i = 0; i < k.length; i++) {
        const curK = k[i];

        curK.modifiers.forEach((mod) => {
          if (!e[`${mod}Key`]) return;
        });

        hotkey = curK;
        break;
      }

      if (hotkey) {
        hotkey.handler({ state: this.state, data: this.data, command: this.command, post: this.post });
      }
    },
  },

  async created() {
    window.addEventListener('message', (event) => {
      const { action, payload } = event.data;

      if (action === 'context-changed') {
        this.context = {
          apiToken: this.context?.apiToken,
          ...payload,
        };
      }
    }, false);

    if (window.parent) {
      window.parent.postMessage({ action: 'init-start' }, "*");
    }

    this.post.throttled = throttle(this.post, 1200);

    try {
      const rawUrl = new URL(this.url);
      let rawIntegrationData = rawUrl.searchParams.get('slyContext');

      this.appUrl = `${rawUrl.origin}${rawUrl.pathname}`;

      let integrationData = {};

      if (rawIntegrationData) {
        try {
          integrationData = JSON.parse(rawIntegrationData);
        } catch (err) {
          console.error(err);
        }
      }

      let pyData = null;
      let stateRes = null;

      if (!integrationData.isStaticVersion) {
        let sessionInfo = null;

        ([sessionInfo = {}, stateRes, pyData] = await Promise.all([
          this.getJson('/sly/session-info'),
          this.getJson('/sly/state', false),
          this.getJson('/sly/data'),
        ]));

        this.sessionInfo = sessionInfo;
      }

      let taskId;
      let apiToken;
      let serverAddress;

      if (localStorage.token) {
        const tokenData = jwtDecode(localStorage.token);
        
        integrationData.apiToken = tokenData.apiToken;
        integrationData.token = localStorage.token;
      }

      if (!window.config) {
        window.config = {};

        if (this.sessionInfo?.ENV || integrationData?.env) {
          window.config = this.sessionInfo?.ENV || integrationData.env;
        }

        Vue.prototype.$env = window.config;
      }

      if (this.sessionInfo?.SERVER_ADDRESS || integrationData?.serverAddress) {
        serverAddress = this.sessionInfo?.SERVER_ADDRESS || integrationData.serverAddress;
      }

      let rawServerAddress = '';

      if (serverAddress) {
        rawServerAddress = serverAddress;

        apiToken = integrationData?.apiToken || this.sessionInfo?.API_TOKEN;
        serverAddress = `${serverAddress.endsWith('/') ? serverAddress.slice(0, -1) : serverAddress}`;

        if (sly.publicApiInstance) {
          sly.publicApiInstance.defaults.baseURL = serverAddress + '/public/api/v3';
          this.publicApiInstance = sly.publicApiInstance;
        } else {
          this.publicApiInstance = axios.create({
            baseURL: `${serverAddress}/public/api/v3`,
          });
        }

        if (sly.apiInstance) {
          sly.apiInstance.defaults.baseURL = serverAddress + '/api';
          this.apiInstance = sly.apiInstance;
        } else {
          this.apiInstance = axios.create({
            baseURL: `${serverAddress}/api`,
          });
        }

        if (localStorage?.token) {
          this.apiInstance.defaults.headers.common.Authorization = `Bearer ${localStorage.token}`;
        }

        if (apiToken) {
          this.context.apiToken = apiToken;
          this.publicApiInstance.defaults.headers.common['x-api-key'] = apiToken;
        }

        taskId = this.sessionInfo?.TASK_ID || integrationData.taskId;

        if (taskId) {
          try {
            const task = await this.publicApiInstance.post('/tasks.info', { id: taskId }).then(r => r.data);
            this.task = task;

            if (this.task?.id) {
              if (sly.apiInstance) {
                this.apiInstance.defaults.headers.common['x-team-id'] = this.task.teamId;
                this.apiInstance.defaults.headers.common['x-workspace-id'] = this.task.workspaceId;
              }

              Object.defineProperties(Vue.prototype, {
                $appSessionTeamId: {
                  get() {
                    return task.teamId;
                  },
                },
              });

              if (integrationData.isStaticVersion) {
                const taskData = this.task?.settings?.customData;

                if (taskData) {
                  const { state = {}, data = {} } = taskData;
                  this.state = state;
                  this.data = data;
                }
              }
            }
          } catch (err) {
            console.error(err);
          }

          if (window.io) {
            if (integrationData.token) {
              connectToSocket(serverAddress);
              this.taskSocket = connectToSocket(serverAddress, 'tasks');

              this.taskSocket.on('changed:progress', this.updateTaskData);
            }
          } else {
            console.warn('socket.io-client isn\'t available');
          }
        }
      }

      this.integrationData = integrationData;

      if (!integrationData.isStaticVersion) {
        let state;

        if (stateRes) {
          this.isDebugMode = !!stateRes.headers.get('x-debug-mode');

          state = await stateRes.json();
        }

        // const data = await this.getJson('/sly/data');

        if (state) {
          this.state = state;
        }

        if (pyData) {
          this.data = pyData;
        }
      }

      if (this.isDebugMode && (serverAddress || (rawServerAddress === '/'))) {
        window.config.SLY_APP_DEBUG_SERVER_ADDRESS = serverAddress || '/';
      }

      document.addEventListener('keypress', this.hotkeysHandler);
    } catch(err) {
      throw err;
    } finally {
      const el = document.querySelector('#app-global-loading-icon');

      if (el) {
        el.style.display = 'none';
      }

      this.loading = false;

      if (window.parent) {
        window.parent.postMessage({ action: 'init-end' }, "*");
      }
    }

    console.log('First Init WS');
    this.connectToWs();
  },

  beforeDestroy() {
    if (this.taskSocket) {
      this.taskSocket.off('changed:progress', this.updateTaskData);
    }

    if (this.wsTimerId) {
      clearInterval(this.wsTimerId);
    }

    document.removeEventListener('keypress', this.hotkeysHandler);
  }
});

window.slyApp = {
  app: null,
  init() {
    if (this.app) return;

    this.app = new Vue({
      el: '#sly-app',
      computed: {
        document() {
          return document;
        }
      },
    });
  },
};

slyApp.init();

// (function initApp() {
//   let interval = setInterval(() => {
//     console.log('----', window.Vue);
//     if (window.Vue) {
//       clearInterval(interval);
//       slyApp.init();
//     }
//   }, 600);
// })()

// let scriptsLoadedCount = 0;
// let domLoaded = false;

// function initApp() {
//   if (!domLoaded || scriptsLoadedCount !== scripts.length) return;
//   slyApp.init();
// }

// document.addEventListener('DOMContentLoaded', function() {
//   domLoaded = true;
//   initApp();
// });

// const scripts = [
//   'https://cdn.jsdelivr.net/npm/jsoneditor@9.7.0/dist/jsoneditor.min.js',
//   'https://cdn.jsdelivr.net/npm/jsoneditor@9.7.0/dist/jsoneditor.min.css',
//   'https://cdn.jsdelivr.net/npm/socket.io-client@2.0.4/dist/socket.io.js',
//   'https://cdn.jsdelivr.net/npm/axios@0.17.1/dist/axios.min.js',
// ];

// scripts.forEach((f) => {
//   let el;
//   let srcField = 'src';

//   if (f.endsWith('.js')) {
//     el = document.createElement('script');

//   } else {
//     srcField = 'href';
//     el = document.createElement('link');
//     el.type = 'text/css';
//     el.rel = 'stylesheet';
//   }

//   el.onload = function () {
//     scriptsLoadedCount += 1;

//     initApp();
//   };

//   el[srcField] = f;

//   document.head.appendChild(el);
// });
