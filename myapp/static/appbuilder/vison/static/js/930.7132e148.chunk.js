"use strict";(self.webpackChunkvite_ml_platform=self.webpackChunkvite_ml_platform||[]).push([[930],{63930:function(e,r,t){t.r(r),t.d(r,{default:function(){return k}});var s=t(31303),n=t(33032),o=t(36222),i=t(18489),a=t(50678),l=t(84322),c=t.n(l),d=t(72791),u=t(23695),f=t(58646),m=t(20135),p=t(66106),_=t(30914),h=t(87309),Z=t(58105),g=t(68747),x=t(79286),v=t(31776),j=t(79271),w=t(80184),y={uuid:Date.now(),getUUid:function(){return this.uuid++}};function k(){var e=(0,j.k6)(),r=d.createRef(),t=(0,d.useState)([{preds:"",uuid:y.getUUid()}]),l=(0,a.Z)(t,2),k=l[0],C=l[1],I=(0,d.useState)([{fe_proc_id:"",fe_proc_name:""}]),b=(0,a.Z)(I,2),N=b[0],P=b[1],F=(0,d.useState)([{model_serv_id:"",model_serv_name:""}]),q=(0,a.Z)(F,2),S=q[0],M=q[1],U=(0,d.useState)({scene_id:"",preds:"",title:"",model_id:"",fe_proc_id:{fe_proc_id:""},resource_info:{model_serv_id:""}}),D=(0,a.Z)(U,2),H=D[0],O=D[1];(0,d.useEffect)((function(){var t={};if(e.location.state){if(t=e.location.state,sessionStorage.setItem("registerModelInformationKeyID",JSON.stringify(t)),O(t),v.Z.featureFeProcConfigDisplayGetQuest(t.scene_id,"fe_proc_config").then((function(e){P(e.result)})),v.Z.featureFeProcConfigDisplayGetQuest(t.scene_id,"model_serv_router").then((function(e){M(e.result)})),"\u66f4\u6539\u6ce8\u518c\u6a21\u578b\u4fe1\u606f"===t.title){var s;null===(s=r.current)||void 0===s||s.setFieldsValue({model_name:t.model_name,model_desc:t.model_desc,owner_rtxs:t.owner_rtxs,resource_info:t.resource_info.model_serv_id,fe_proc_id:t.fe_proc_id.fe_proc_name});var n;n=t.preds.split(",").map((function(e,r){return{preds:e}})),C(t.preds.split(",")?n:[])}}else if(t=JSON.parse(sessionStorage.getItem("registerModelInformationKeyID")||""),O(t),v.Z.featureFeProcConfigDisplayGetQuest(t.scene_id,"fe_proc_config").then((function(e){P(e.result)})),v.Z.featureFeProcConfigDisplayGetQuest(t.scene_id,"model_serv_router").then((function(e){M(e.result)})),"\u66f4\u6539\u6ce8\u518c\u6a21\u578b\u4fe1\u606f"===t.title){var o;null===(o=r.current)||void 0===o||o.setFieldsValue({model_name:t.model_name,model_desc:t.model_desc,owner_rtxs:t.owner_rtxs,resource_info:t.resource_info.model_serv_id,fe_proc_id:t.fe_proc_id.fe_proc_name});var i;i=t.preds.split(",").map((function(e,r){return{preds:e}})),C(t.preds.split(",")?i:[])}}),[]),(0,d.useEffect)((function(){for(var e=0;e<k.length;e++){var t;null===(t=r.current)||void 0===t||t.setFieldsValue((0,o.Z)({},"preds".concat(e),k[e].preds))}}),[k]);var Q=function(){var e=(0,n.Z)(c().mark((function e(){var t;return c().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,new Promise((function(e,t){try{var s,n=null===(s=r.current)||void 0===s?void 0:s.getFieldsValue(),o=/(\d+)$/,l=[],c={};Object.entries(n).forEach((function(e){var r=(0,a.Z)(e,2),t=r[0],s=r[1];if(o.test(t)){var n=RegExp.$1;l[n]||(l[n]={}),l[n][t.substring(0,t.length-1)]=s}else c[t]=s})),e(l.map((function(e){return(0,i.Z)((0,i.Z)({},e),{},{uuid:y.getUUid()})})))}catch(d){t(d)}}));case 3:t=e.sent,C([].concat((0,s.Z)(t),[{preds:"",uuid:y.getUUid()}])),e.next=10;break;case 7:e.prev=7,e.t0=e.catch(0),console.error(e.t0);case 10:case"end":return e.stop()}}),e,null,[[0,7]])})));return function(){return e.apply(this,arguments)}}();return(0,w.jsx)("div",{className:"RegisterModelInformationClass",children:(0,w.jsxs)("div",{className:"bodyClass",children:[(0,w.jsx)("div",{className:"SceneHeader",children:"\u6ce8\u518c\u6a21\u578b\u4fe1\u606f"}),(0,w.jsxs)(f.Z,{name:"basic",labelCol:{span:8},wrapperCol:{span:16},onFinish:function(r){var t={},s=[];for(var n in r)n.includes("preds")?s.push(r[n]):t[n]=r[n];t.preds=s.toString(),"\u66f4\u6539\u6ce8\u518c\u6a21\u578b\u4fe1\u606f"===H.title?v.Z.featureUpdateModelConfigPostQuest((0,i.Z)((0,i.Z)({},t),{},{scene_id:H.scene_id,fe_proc_id:H.fe_proc_id.fe_proc_id,model_id:H.model_id})).then((function(r){0===r.retcode?e.push("/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation"):r.retmsg&&u.ZP.error("\u5931\u8d25\u539f\u56e0: ".concat(r.retmsg))})):v.Z.featureRegisterModelConfigPostQuest((0,i.Z)((0,i.Z)({},t),{},{scene_id:H.scene_id})).then((function(r){0===r.retcode?e.push("/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation"):r.retmsg&&u.ZP.error("\u5931\u8d25\u539f\u56e0: ".concat(r.retmsg))}))},onFinishFailed:function(e){console.log("Failed:",e)},autoComplete:"off",ref:r,children:[(0,w.jsx)(f.Z.Item,{label:"\u6a21\u578b\u4e2d\u6587\u540d",name:"model_name",rules:[{required:!0,message:"\u8bf7\u8f93\u5165\u6a21\u578b\u4e2d\u6587\u540d"}],children:(0,w.jsx)(m.Z,{style:{width:160}})}),(0,w.jsx)(f.Z.Item,{label:"\u6a21\u578b\u82f1\u6587\u540d",name:"model_desc",rules:[{required:!0,message:"\u8bf7\u8f93\u5165\u6a21\u578b\u82f1\u6587\u540d"}],children:(0,w.jsx)(m.Z,{style:{width:160}})}),(0,w.jsx)(f.Z.Item,{label:"\u9884\u6d4b\u76ee\u6807",children:k.map((function(e,r){return(0,w.jsxs)(p.Z,{gutter:16,children:[(0,w.jsx)(_.Z,{className:"gutter-row",span:16,children:(0,w.jsx)(f.Z.Item,{name:"preds".concat(r),rules:[{required:!0,message:"\u8bf7\u8f93\u5165"}],children:(0,w.jsx)(m.Z,{style:{width:160}})})}),(0,w.jsx)(_.Z,{className:"gutter-row",span:4,children:(0,w.jsx)(f.Z.Item,{children:(0,w.jsx)(h.Z,{className:"ButtonClass",type:"dashed",onClick:function(){return(r=e).uuid&&C(k.filter((function(e){return e.uuid!==r.uuid}))),void(r.uuid||C(k.filter((function(e){return e!==r}))));var r},block:!0,icon:(0,w.jsx)(g.Z,{})})})}),(0,w.jsx)(_.Z,{className:"gutter-row",span:4,children:(0,w.jsx)(f.Z.Item,{children:(0,w.jsx)(h.Z,{className:"ButtonClass",type:"dashed",onClick:function(){return Q()},block:!0,icon:(0,w.jsx)(x.Z,{})})})})]},e.uuid)}))}),(0,w.jsx)(f.Z.Item,{hasFeedback:!0,label:"\u7279\u5f81\u63d2\u4ef6\u914d\u7f6e",children:(0,w.jsxs)(p.Z,{gutter:10,children:[(0,w.jsx)(_.Z,{className:"gutter-row",span:16,children:(0,w.jsx)(f.Z.Item,{name:"fe_proc_id",rules:[{required:!0,message:"\u8bf7\u8f93\u5165\u7279\u5f81\u63d2\u4ef6\u914d\u7f6e"}],children:(0,w.jsx)(Z.Z,{placeholder:"Please select a country",style:{width:160},children:N.map((function(e){return(0,w.jsx)(Z.Z.Option,{value:e.fe_proc_id,children:e.fe_proc_name},e.fe_proc_id)}))})})}),(0,w.jsx)(_.Z,{className:"gutter-row",span:4,children:(0,w.jsx)(f.Z.Item,{children:(0,w.jsx)(h.Z,{onClick:function(){e.push({pathname:"/HeterogeneousPlatform/Nationalkaraoke/FeatureConfiguration",state:(0,i.Z)((0,i.Z)({},H),{},{title:"\u7279\u5f81\u63d2\u4ef6\u914d\u7f6e"})})},className:"ButtonClass",type:"dashed",block:!0,icon:(0,w.jsx)(x.Z,{})})})})]})}),(0,w.jsx)(f.Z.Item,{label:"\u6a21\u578b\u670d\u52a1\u4fe1\u606f",hasFeedback:!0,children:(0,w.jsxs)(p.Z,{gutter:16,children:[(0,w.jsx)(_.Z,{className:"gutter-row",span:16,children:(0,w.jsx)(f.Z.Item,{name:"resource_info",rules:[{required:!0,message:"\u8bf7\u8f93\u5165\u6a21\u578b\u670d\u52a1\u4fe1\u606f"}],children:(0,w.jsx)(Z.Z,{placeholder:"Please select a country",style:{width:160},children:S.map((function(e){return(0,w.jsx)(Z.Z.Option,{value:e.model_serv_id,children:e.model_serv_name},e.model_serv_id)}))})})}),(0,w.jsx)(_.Z,{className:"gutter-row",span:4,children:(0,w.jsx)(f.Z.Item,{children:(0,w.jsx)(h.Z,{onClick:function(){e.push("/HeterogeneousPlatform/Nationalkaraoke/RegisterModelService")},className:"ButtonClass",type:"dashed",block:!0,icon:(0,w.jsx)(x.Z,{})})})})]})}),(0,w.jsx)(f.Z.Item,{label:"\u8d23\u4efb\u4eba",name:"owner_rtxs",rules:[{required:!0,message:"\u8bf7\u8f93\u5165\u8d23\u4efb\u4eba"}],children:(0,w.jsx)(m.Z,{style:{width:160}})}),(0,w.jsx)(f.Z.Item,{wrapperCol:{offset:8,span:16},children:(0,w.jsx)(h.Z,{className:"preservationClass",type:"primary",htmlType:"submit",children:"\u4fdd\u5b58"})})]})]})})}},79286:function(e,r,t){t.d(r,{Z:function(){return l}});var s=t(1413),n=t(72791),o={icon:{tag:"svg",attrs:{viewBox:"64 64 896 896",focusable:"false"},children:[{tag:"defs",attrs:{},children:[{tag:"style",attrs:{}}]},{tag:"path",attrs:{d:"M482 152h60q8 0 8 8v704q0 8-8 8h-60q-8 0-8-8V160q0-8 8-8z"}},{tag:"path",attrs:{d:"M176 474h672q8 0 8 8v60q0 8-8 8H176q-8 0-8-8v-60q0-8 8-8z"}}]},name:"plus",theme:"outlined"},i=t(54291),a=function(e,r){return n.createElement(i.Z,(0,s.Z)((0,s.Z)({},e),{},{ref:r,icon:o}))};a.displayName="PlusOutlined";var l=n.forwardRef(a)}}]);