<div class="experiment-body">
    <sly-style>
.experiment-body .sly-markdown-widget .markdown-body {
    padding: 0;
    font-family: inherit;
}

.experiment-body .sly-markdown-widget .markdown-body h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 0px;
    border: 0;
}

.experiment-body .sly-markdown-widget .markdown-body h3 {
    color: #949bab;
    font-size: 18px;
    margin-bottom: 7px;
}

.experiment-body .sly-markdown-widget .markdown-body p {
    margin-bottom: 12px;
}

.experiment-body .section {
    margin-bottom: 30px;
    scroll-margin-top: 20px;
    padding-top: 10px;
}

.experiment-body .overview-info-block {
    background: #f4f7fb;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

.experiment-body .width-fit-content {
    width: fit-content;
}

.experiment-body .main-content {
    flex-grow: 1;
    margin-right: 20px;
}

.experiment-body .buttons-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 20px;
}

.experiment-body hr {
    display: none
}

.experiment-body blockquote {
    margin: 15px 0 0;
}

.experiment-body table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}

.experiment-body table th,
.experiment-body table td {
    border: 1px solid #e0e0e0;
    padding: 8px 12px;
    text-align: left;
}

.experiment-body table th {
    background-color: #f4f7fb;
}

.experiment-body .table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
    /* box-shadow: 0 2px 6px rgba(0,0,0,0.05); */
    border-radius: 6px;
    overflow: hidden;
}

.experiment-body .table th {
    background-color: #f4f7fb;
    font-weight: 600;
    color: #333;
    padding: 12px 15px;
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
}

.experiment-body .table td {
    padding: 10px 15px;
    border-bottom: 1px solid #e9ecef;
}

.experiment-body .table tr:hover {
    background-color: #f8f9fa;
}

.experiment-body .links-list,
.experiment-body .overview-list,
.experiment-body .buttons-list {
    list-style-type: none;
    padding: 0;
    margin: 0 0 20px 0;
}

.experiment-body .links-list li,
.experiment-body .overview-list li,
.experiment-body .buttons-list li {
    margin-bottom: 8px;
    position: relative;
    padding-left: 8px;
    display: flex;
    align-items: flex-start;
}

.experiment-body .links-list li:before,
.experiment-body .buttons-list li:before {
    content: "•";
    color: #1890ff;
    display: inline-block;
    width: 15px;
    margin-right: 10px;
}

.experiment-body .overview-list li:before {
    content: "";
    display: none;
}

.experiment-body .link-icon,
.experiment-body .button-icon {
    margin-right: 8px;
}

.experiment-body .overview-list .label {
    font-weight: 600;
    margin-right: 8px;
    min-width: 120px;
    display: inline-block;
}

/* Code */
.experiment-body .code-block {
    background-color: #f5f5f5;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 20px;
    overflow-x: auto;
}

.experiment-body details {
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 20px;
}

.experiment-body details summary {
    padding: 12px 15px;
    background-color: #f4f7fb;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: background-color 0.2s;
    width: fit-content;
}

.experiment-body details summary:hover {
    background-color: #e6edf7;
}

.experiment-body details[open] summary {
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
    margin-bottom: 10px;
}

/* Classes overview styles */
.classes-overview {
    margin-bottom: 30px;
}

.class-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.class-tag {
    background: #e3e8ef;
    color: #434a53;
    padding: 3px 7px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 400;
}


.experiment-main-wrapper {
    display: flex;
}

.experiment-main-content .main-content.markdown-body {
    font-size: 14px;
    font-family: inherit;
}

.experiment-main-content .main-content.markdown-body h2 {
    font-size: 18px;
    font-weight: 700;
    /* border: 0;
    padding: 0; */
}

.experiment-main-content .main-content.markdown-body h3 {
    font-size: 16px;
    font-weight: 700;
    border: 0;
}

.experiment-main-content .main-content.markdown-body .img-grid-gallery-v2-header {
    font-size: 13px;
}

/*********************** Header ***********************/

.experiment-body .experiment-info-header {
    background: #f8fafc;
    border-radius: 6px;
    font-size: 14px;
}

.experiment-body .experiment-info-header h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    padding: 0;
    border: 0;
}

.experiment-body .experiment-info-header ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.experiment-body .experiment-info-header .experiment-info-subheader {
    display: flex;
    background: #f1f5f9;
    padding: 15px 20px;
    justify-content: space-between;
    align-items: center;
    border-radius: 6px 6px 0 0;
}

.experiment-body .experiment-info-header .experiment-info-subheader-body {
    padding: 15px 20px;
    display: flex;
    gap: 30px;
}

.experiment-body .experiment-info-header .experiment-info-subheader-body h2 {
    border: 0;
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    margin-right: 5px;
}

.experiment-body .experiment-info-header .experiment-info-subheader-body ul strong {
    width: 100px;
}

.experiment-body .experiment-info-header .experiment-info-subheader-block {
    padding-right: 30px;
    border-right: 1px solid #dde4ef;
}

/*.experiment-body .experiment-info-header .experiment-info-subheader-block:nth-last-child(2) { border: none; flex: 1; }*/

.experiment-body .experiment-info-header .experiment-info-subheader-block:last-child {
    border: none;
}

.experiment-body .experiment-info-header .experiment-info-subheader-block .experiment-info-subheader-mini-title {
    display: flex;
    align-items: center;
    gap: 3px;
    margin: 0 0 6px;
}

.experiment-body .experiment-info-header .experiment-info-subheader-block .experiment-info-subheader-mini-title a {
    padding: 2px 6px;
    border: 1px solid #ccd3d9;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    background: white;
    /*color: #1a6cdb;*/
    color: black;
    margin-left: 3px;
    margin-top: -2px;
}

.experiment-body .experiment-info-header .experiment-info-buttons {
    display: flex;
    gap: 5px;
    align-items: center;
}

.experiment-body .experiment-info-header .experiment-info-buttons a, .training-plots-button {
    display: flex;
    gap: 5px;
    align-items: center;
    background: white;
    padding: 7px 13px;
    border: 1px solid #ccd3d9;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
}


.experiment-body .experiment-info-header .experiment-info-buttons a.primary, .training-plots-button {
    background: black;
    color: white;
}

.experiment-body .experiment-info-header .experiment-info-buttons img {
    height: 16px;
    width: auto;
    background: transparent;
}

.experiment-body .training-plots-button {
    display: inline-block;
}

/*********************** Metrics ***********************/

.experiment-body .experiment-metric {
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.experiment-body .experiment-metric .experiment-label {
    width: 100px;
    font-weight: 600;
}

.experiment-body .experiment-metric .experiment-value {
    margin-right: 5px;
    text-align: right;
}

.experiment-body .experiment-progress-bar {
    background: #eee;
    height: 6px;
    border-radius: 6px;
    overflow: hidden;
    width: 90px;
}

.experiment-body .experiment-progress {
    height: 100%;
    background: #28a745;
}

.experiment-body .experiment-info-header .experiment-info-subheader-footer-menu {
    display: flex;
    gap: 5px;
    margin: 0px 20px;
    padding-top: 5px;
    border-top: 1px solid #dde4ef;
    font-size: 14px;
}

.experiment-body .experiment-info-header .experiment-info-subheader-footer-menu a {
    line-height: 36px;
    padding: 0 7px;
    color: black;
    border-bottom: 2px solid transparent;
}

.experiment-body .experiment-info-header .experiment-info-subheader-footer-menu a:hover {
    text-decoration: none;
}

.experiment-body .experiment-info-header .experiment-info-subheader-footer-menu a.active {
    border-color: #20a0ff;
    font-weight: 500;
    color: #20a0ff;
}    </sly-style>

    <div class="experiment-main-wrapper">
        <div class="experiment-main-content">
            <div class="main-content markdown-body">
<div class="experiment-info-header">
    <div class="experiment-info-subheader">
        <h1>Online Training - Task 54169</h1>

        <div class="experiment-info-buttons">
            <!-- Download -->
            <sly-iw-download-button
                style="padding: 7px 8px;"
                :disabled="false"
                teamFilesPath="/experiments/4596/100001_online_training_Mask2Former/"   
                :command="command">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAABn0lEQVR4nO2YT04CMRSHPzdzDEAT4lZIvICDHsiFia7Y4BEQTsF50CiCW42iCzeTjKkpyXNScDodp4P0S17Sgfb1/fr6ZzoQCAQC/5VDYAE8Am22kCsg1XbJFtIXAlR56+gHAZ4JGfBNyIBvQgbKogWMgScgEXu7tE9gYJGBa90mNVii+xoBTdfgz4D3NR1l7SOngD1dN4/PJdBzGfm8wdtmYLAhA6lBRKOIgLFw8gCcAFEFizgCYmAm2t8UiP97Hq4cqOCr3oVi0X5eoP2PBWsz8mUJiDIL2xo5D32dA6lLDDsj4AC4A+6Bbg4BXV1Xzet2HQSci3ovQoRJQFfXWf1+UQcBrUxQqtwxCOgY6u3XQYBpZFV5Ip4nhv/ldPMuQHEEPOc4WV+B4z+KwbnxbyJsgve2ja4TYRu813MgK6JI8N4PMrVIb4FpzgVbOwFlkO60gMTxbdSVyPVtdOF4H3ClJ/pXn+itGQkHM+2wikxEwGnmRjYs4qip76OpZ3sreidGj/rSc/AxjjR0CucbvguVaYnua+gy8oFAIEAlfAG3TJLaWbXlEAAAAABJRU5ErkJggg==" alt="download--v1">
            </sly-iw-download-button>
            <!-- Finetune -->
            <sly-iw-launch-button
                style=""
                :disabled="false"
                :autoRun="false"
                :openInNewWindow="true"
                :selectContext="false"
                :state="{ 'selectedExperimentTaskId': , 'mode': 'finetune' }"
                :moduleId="None"
                :command="command">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAACeElEQVR4nO2Yy2oVQRCGv+AhUXyBqDFqHiEighIXgpHcFH0EF3GnWfgICaJuzEbNwn127oIuYrbeLyAuIiQcycWdRkFNEEZKaqBoZsY5mdMzc6B/aA4zPaf6q+6q6p6BoKCgoJzqAUaAOeA1sAH8Ab4C74GHwGWgm5qpAVxV4ChH2wTGqYn6gTc5wSPTZEUq1wngiwO2CswAp4ADGi7i5JLz3N2q4fsd+J/AVEJ8dwGzDvy8hl1lajhhs6mrQSfAowlrZ74IfEPD6QMwQUml0labqYLw885Kei+xo07CJg14ZxfwkbZLvh2YM4NNJ/RPFICPdLPzqldmsNMJ/Yum/1FOeFti3/l2wMa/1HlX303/wRzwkitHzPU33w5sm8EkoV1tpTjQSIHvUjvxvZ0yV+DQf0JoAegDDms4JcGjjsb313078NwMNpTQP57jHDRr4FE7cd9L3w7YEnkr5ZnpDPjbDrzopum/59uBa2awtxnPjQFPgR/aFnUPcbUXaBqb5z2y/6vxNomftMHmDWNsupervisely/template/experiment/header.html.jinja..." alt="support">
                <span>Finetune</span>
            </sly-iw-launch-button>

            <!-- Continue -->
            <sly-iw-launch-button
                style=""
                :disabled="false"
                :autoRun="false"
                :openInNewWindow="true"
                :selectContext="false"
                :state="{ 'selectedExperimentTaskId': , 'mode': 'continue' }"
                :moduleId="None"
                :command="command">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAYAAADimHc4AAAACXBIWXMAAAsTAAALEwEAmpwYAAAFcklEQVR4nO2da4hVVRTHf42Zjjg1RUwmSWX0GqlMe9sDUokKv0RUFvVBoje9vkT6oaDI3mVR9KAPFUUY5IfACS2bqKAHOVMRKFpN1JSNWc1ILytPbPgPXS5z7z1733vOvufe9YMNcuc81l7rnL3/55y1tmAYhmEYhmEYhmEYhmEYzccewBnASuAtYBgYUxvWbyu1jdvWaBDTgOuBzUCSsm3WPm5fI5C9gZuA7z0cX962A3cC+1kU0nOAnPZrHY4vbzuBVcAsC0RlDpOT/mig48vbLuAFoNcC8T/Hyyn/ZOj48rYbeB04tV0D4VTKuUB/gPMGgVt1FXeq9eq3wYDj9cuWtlBOHcAS4KCAR72nfWs56nRd3bs9j/8ZcAWwJy3IFHXOR0omwL9y5skB55yroe1vz3N+LfU1rZWk5LCnE/6S845qgA2zNbn/7mnDSJEl7LiU/MWz02Ny1kEZ2NQjm35uZQkbKiV/lHP2zcHGLt2V3wXelb2tJCW/kjOcksmbvTQvbQqcl5pCwo4rjsSzDTaR4uiQuvowQ2VWOClZtAvq0zwuqHql5CkUh7mBQ2omEbarTil5NMVldkwJGyold2YoJWPRk6cvnDp4VFewz8m2AbcD3eRDJ/AUMKr2bIMe3KrRrT5uCxgNHtMwXpOHPQ++FbgGmEq+PF1hvlmTw3wzVX3e6ukrd2HX5KeUB9sIXAxMIg6/1bDvHeD8jBXXJPlgY0qfuafwlghAh15bpLHzc+ByYHITBMB9Xq3JQwUZgh73tPMb4GZgesQh6N60k/AjBZiEp8jOPz3t3AHcJWWT1yTsbLzP9y6cCTzgcauPt1Gd7EDyYSZwv87rY6fT9k9I69fCPeHeE3COUdnmbAzGRX15YNSfAQ4nH/YBbgtIY3FPu68A82oc/zi9cog2GjjdfS3wpWcHnTR8FTiB/IamKwNembi2DlhU463Ax7HnQzfzXwIMBHTwTWAx+SmlC4APAuz8BLiogsI7eIJXE9EU4TnAhgZ3MAvOAtYGfKh3V/SyCY63Sn/fIB9E5yTgNQ03Ph3cAlyV9hG9ARwLvKgELR87XcpLKUeoz03HkQFZCAnwgyZQN5FmjTvHk572uTfBhSGpo43qYWVGBnbN0LF9pWSiO7stApBkkJZyaOA7/fJWGCp14Dp9hPfV6KuB+QF2zNe+vvNSywZg/IlyaWAO53pgYYrzL9S2vscflG0tHYBGJOcOSJXMKUnOnaPfBhqQnNsWAWjm9PSk3QLQbAUabRuAZilRavsAxC7SswBUKFP1yeHcVEeZqgWgAk6lLADulswcKksupervisely/template/experiment/header.html.jinja..." alt="graduation-cap">
                <span>Continue</span>
            </sly-iw-launch-button>

            <!-- From Scratch -->
            <sly-iw-launch-button
                style=""
                :disabled="false"
                :autoRun="false"
                :openInNewWindow="true"
                :selectContext="true"
                :state="{ 'mode': 'from_scratch' }"
                :moduleId="None"
                :command="command">
                <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAYAAABXAvmHAAAACXBIWXMAAAsTAAALEwEAmpwYAAAFcklEQVR4nO2da4hVVRTHf42Zjjg1RUwmSWX0GqlMe9sDUokKv0RUFvVBoje9vkT6oaDI3mVR9KAPFUUY5IfACS2bqKAHOVMRKFpN1JSNWc1ILytPbPgPXS5z7z1733vOvufe9YMNcuc81l7rnL3/55y1tmAYhmEYhmEYhmEYhmEYzccewBnASuAtYBgYUxvWbyu1jdvWaBDTgOuBzUCSsm3WPm5fI5C9gZuA7z0cX962A3cC+1kU0nOAnPZrHY4vbzuBVcAsC0RlDpOT/mig48vbLuAFoNcC8T/Hyyn/ZOj48rYbeB04tV0D4VTKuUB/gPMGgVt1FXeq9eq3wYDj9cuWtlBOHcAS4KCAR72nfWs56nRd3bs9j/8ZcAWwJy3IFHXOR0omwL9y5skB55yroe1vz3N+LfU1rZWk5LCnE/6S845qgA2zNbn/7mnDSJEl7LiU/MWz02Ny1kEZ2NQjm35uZQkbKiV/lHP2zcHGLt2V3wXelb2tJCW/kjOcksmbvTQvbQqcl5pCwo4rjsSzDTaR4uiQuvowQ2VWOClZtAvq0zwuqHql5CkUh7mBQ2omEbarTil5NMVldkwJGyold2YoJWPRk6cvnDp4VFewz8m2AbcD3eRDJ/AUMKr2bIMe3KrRrT5uCxgNHtMwXpOHPQ++FbgGmEq+PF1hvlmTw3wzVX3e6ukrd2HX5KeUB9sIXAxMIg6/1bDvHeD8jBXXJPlgY0qfuafwlghAh15bpLHzc+ByYHITBMB9Xq3JQwUZgh73tPMb4GZgesQh6N60k/AjBZiEp8jOPz3t3AHcJWWT1yTsbLzP9y6cCTzgcauPt1Gd7EDyYSZwv87rY6fT9k9I69fCPeHeE3COUdnmbAzGRX15YNSfAQ4nH/YBbgtIY3FPu68A82oc/zi9cog2GjjdfS3wpWcHnTR8FTiB/IamKwNembi2DlhU463Ax7HnQzfzXwIMBHTwTWAx+SmlC4APAuz8BLiogsI7eIJXE9EU4TnAhgZ3MAvOAtYGfKh3V/SyCY63Sn/fIB9E5yTgNQ03Ph3cAlyV9hG9ARwLvKgELR87XcpLKUeoz03HkQFZCAnwgyZQN5FmjTvHk572uTfBhSGpo43qYWVGBnbN0LF9pWSiO7stApBkkJZyaOA7/fJWGCp14Dp9hPfV6KuB+QF2zNe+vvNSywZg/IlyaWAO53pgYYrzL9S2vscflG0tHYBGJOcOSJXMKUnOnaPfBhqQnNsWAWjm9PSk3QLQbAUabRuAZilRavsAxC7SswBUKFP1yeHcVEeZqgWgAk6lLADulswcKksupervisely/template/experiment/header.html.jinja..." alt="new">
                <span>From Scratch</span>
            </sly-iw-launch-button>
        </div>
    </div>
    <div class="experiment-info-subheader-body">
        <div class="experiment-info-subheader-block">
            <div class="experiment-info-subheader-mini-title">
                <h2 id="training-info">Training Info</h2>
                <a href="https://dev.internal.supervisely.com/apps/sessions/54169" target="_blank">Session <i class="zmdi zmdi-arrow-right-top" /></a>
                <a href="https://dev.internal.supervisely.com/files/?path=/experiments/4596/100001_online_training_Mask2Former/logs" target="_blank">Logs <i class="zmdi zmdi-arrow-right-top" /></a>
            </div>
            <ul>
            <li><strong>Session ID:</strong> 1765363422</li>
            <li><strong>Date:</strong> 2025-12-10 10:43:42</li>
            <li><strong>Duration:</strong> 0h 13m</li>
            <li><strong>Device:</strong> NVIDIA GeForce RTX 4090</li>
            <li><strong>Total Iterations:</strong> 680</li>
            </ul>
        </div>
        <div class="experiment-info-subheader-block">
            <div class="experiment-info-subheader-mini-title">
                <h2 id="model">Model</h2>
            </div>
            <ul>
            <li><strong>Architecture:</strong> Mask2Former</li>
            <li><strong>Backbone:</strong> SwinTransformer</li>
            <li><strong>Task:</strong> semantic segmentation</li>
            <li><strong>Classes:</strong> (10) </li>
            </ul>
        </div>
        <div class="experiment-info-subheader-block">
            <div class="experiment-info-subheader-mini-title">
                <h2 id="training-data">Training Data</h2>
                <a href="https://dev.internal.supervisely.com/projects/4596/datasets" target="_blank">Project <i class="zmdi zmdi-arrow-right-top" /></a>
                <a href="https://dev.internal.supervisely.com/files/?path=/experiments/4596/100001_online_training_Mask2Former" target="_blank">Artifacts <i class="zmdi zmdi-arrow-right-top" /></a>
            </div>
            <ul>
            <li><strong>Project:</strong> tomatoes_test</li>
            <li><strong>Final Size:</strong> 2 samples</li>
            <li><strong>Initial Samples:</strong> 2</li>
            <li><strong>Samples Added:</strong> 0</li>
            </ul>
        </div>
    </div>
    <div class="experiment-info-subheader-footer-menu">
        <a :href="`${location.pathname}#training-results`" class="active">Training Results</a>
        <a :href="`${location.pathname}#training-plots`">Training Plots</a>
        <a :href="`${location.pathname}#checkpoints`">Checkpoints</a>
        <a :href="`${location.pathname}#hyperparameters`">Hyperparameters</a>
        <a :href="`${location.pathname}#configuration`">Configuration</a>
    </div>
</div>                
<h2 id="about-live-training">About Live Training</h2>
<hr />
<h2 id="training-results">Training Results</h2>
<p><sly-iw-image src="/data/training_plots_grid.png" :template-base-path="templateBasePath" :options="{ style: { width: '70%', height: 'auto' } }" /></p>
<h2 id="checkpoints">Checkpoints</h2>
<p><a href="https://dev.internal.supervisely.com/files/?path=/experiments/4596/100001_online_training_Mask2Former" target="_blank">📂 Open in Team Files</a></p>
<details>
    <summary>📋 Checkpoints</summary>

<table class="table">
<thead><tr><th>Checkpoint Name</th><th>Iteration</th><th>Loss</th><th>Actions</th></tr></thead>
<tbody>
<tr><td>plateau_iter_689.pth</td><td>689</td><td>3.403881</td><td><a href="https://dev.internal.supervisely.com/files/?path=/experiments/4596/100001_online_training_Mask2Former/checkpoints/plateau_iter_689.pth" target="_blank" class="download-link">Download</a></td></tr>
<tr><td>plateau_iter_759.pth</td><td>759</td><td>N/A</td><td><a href="https://dev.internal.supervisely.com/files/?path=/experiments/4596/100001_online_training_Mask2Former/checkpoints/plateau_iter_759.pth" target="_blank" class="download-link">Download</a></td></tr>
</tbody>
</table>

</details>

<h2 id="classes">Classes</h2>
<details>
    <summary>📝 Classes</summary>

The model can predict 10 classes.

| Class Name |
|------------|
| Columella |
| Core |
| Locule |
| ... |

</details>            </div>
        </div>
    </div>
</div>