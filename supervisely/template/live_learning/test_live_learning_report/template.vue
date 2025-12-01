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
        <div class="experiment-info-header-title">
            <h1>Live Learning</h1>
        </div>
        <div class="experiment-info-buttons">
            <a href="https://dev.internal.supervisely.com/files/?path=/live-learning/4709_test_project/1234568911" target="_blank">
                <i class="zmdi zmdi-folder"></i> Artifacts
            </a>
            <a href="https://dev.internal.supervisely.com/projects/4709/datasets" target="_blank">
                <i class="zmdi zmdi-image"></i> Project
            </a>
        </div>
    </div>
    
    <div class="experiment-info-subheader-body">
        <div class="experiment-info-subheader-block">
            <h2 id="session-info">Session Info</h2>
            <ul>
                <li><strong>Session ID:</strong> 1234568911</li>
                <li><strong>Started:</strong> 2025-12-01 15:49:44</li>
                <li><strong>Duration:</strong> 2h 15m</li>
                <li><strong>Total Iterations:</strong> 150</li>
            </ul>
        </div>
        
        <div class="experiment-info-subheader-block">
            <h2 id="model">Model</h2>
            <ul>
                <li><strong>Architecture:</strong> Mask2Former</li>
                <li><strong>Backbone:</strong> Swin-T</li>
                <li><strong>Classes:</strong> (3) <span class="class-tag">car</span> <span class="class-tag">person</span> <span class="class-tag">tree</span> </li>
            </ul>
        </div>
        
        <div class="experiment-info-subheader-block">
            <h2 id="dataset">Dataset</h2>
            <ul>
                <li><strong>Project:</strong> nauplius2048</li>
                <li><strong>Final Size:</strong> 0 samples</li>
                <li><strong>Initial Samples:</strong> 0</li>
                <li><strong>Samples Added:</strong> 0</li>
            </ul>
        </div>
    </div>
</div>                
<h2 id="training-results">Training Results</h2>
<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>                        <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-3.3.0.min.js" integrity="sha256-bO3dS6yCpk9aK4gUpNELtCiDeSYvGYnK7jFI58NQnHI=" crossorigin="anonymous"></script>                <div id="training-plot" class="plotly-graph-div" style="height:400px; width:100%;"></div>            <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                if (document.getElementById("training-plot")) {                    Plotly.newPlot(                        "training-plot",                        [{"line":{"color":"#1890ff","width":2},"mode":"lines","name":"Train Loss","x":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],"y":[0.9235056600070435,0.8543255500777847,0.7917602987106243,0.7351774068557442,0.6840048537100292,0.637725313912609,0.5958709276801656,0.5580185710110486,0.523785578142659,0.4928258730185834,0.4648264706568111,0.43950431304986504,0.41660340760959524,0.3958922392279436,0.37716142979108874],"type":"scatter"}],                        {"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"#C8D4E3","linecolor":"#C8D4E3","minorgridcolor":"#C8D4E3","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scattermap":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermap"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"#C8D4E3"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""},"bgcolor":"white","radialaxis":{"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"yaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"},"zaxis":{"backgroundcolor":"white","gridcolor":"#DFE8F3","gridwidth":2,"linecolor":"#EBF0F8","showbackground":true,"ticks":"","zerolinecolor":"#EBF0F8"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"baxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""},"bgcolor":"white","caxis":{"gridcolor":"#DFE8F3","linecolor":"#A2B1C6","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"#EBF0F8","linecolor":"#EBF0F8","ticks":"","title":{"standoff":15},"zerolinecolor":"#EBF0F8","zerolinewidth":2}}},"margin":{"l":50,"r":50,"t":50,"b":50},"title":{"text":"Training Loss"},"xaxis":{"title":{"text":"Iteration"}},"yaxis":{"title":{"text":"Loss"}},"height":400},                        {"responsive": true}                    )                };            </script>        </div>
</body>
</html>

<h2 id="checkpoints">Checkpoints</h2>
<table class="table">
<thead><tr><th>Checkpoint Name</th><th>Iteration</th><th>Loss</th><th>Actions</th></tr></thead>
<tbody>
<tr><td>checkpoint_50.pth</td><td>50</td><td>0.456000</td><td><a href="https://dev.internal.supervisely.com/files/?path=/live-learning/4709_test_project/1234568911/checkpoints/checkpoint_50.pth" target="_blank" class="download-link">Download</a></td></tr>
<tr><td>checkpoint_100.pth</td><td>100</td><td>0.312000</td><td><a href="https://dev.internal.supervisely.com/files/?path=/live-learning/4709_test_project/1234568911/checkpoints/checkpoint_100.pth" target="_blank" class="download-link">Download</a></td></tr>
<tr><td>checkpoint_150.pth</td><td>150</td><td>0.234000</td><td><a href="https://dev.internal.supervisely.com/files/?path=/live-learning/4709_test_project/1234568911/checkpoints/checkpoint_150.pth" target="_blank" class="download-link">Download</a></td></tr>
</tbody>
</table>

<h2 id="hyperparameters">Hyperparameters</h2>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>learning_rate</td>
<td>0.0002</td>
</tr>
<tr>
<td>batch_size</td>
<td>2</td>
</tr>
<tr>
<td>crop_size</td>
<td>(768, 768)</td>
</tr>
<tr>
<td>max_epochs</td>
<td>100000</td>
</tr>
<tr>
<td>weight_decay</td>
<td>0.0005</td>
</tr>
<tr>
<td>optimizer</td>
<td>AdamW</td>
</tr>
</tbody>
</table>
<h2 id="configuration">Configuration</h2>
<p><strong>Config file:</strong> <code>configs/mask2former/custom/swin-t-online-app.py</code></p>            </div>
        </div>
    </div>
</div>