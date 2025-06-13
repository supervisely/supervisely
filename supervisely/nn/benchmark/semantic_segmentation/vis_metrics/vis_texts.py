markdown_header = """
<h1>{}</h1>

<div class="model-info-block">
    <div>Created by <b>{}</b></div>
    <div><i class="zmdi zmdi-calendar-alt"></i><span>{}</span></div>
</div>
"""

markdown_overview = """
- **Model**: {}
- **Checkpoint**: {}
- **Architecture**: {}
- **Task type**: {}
- **Runtime**: {}
- **Checkpoint file**: <a href="{}" target="_blank">{}</a>
- **Ground Truth project**: <a href="/projects/{}/datasets" target="_blank">{}</a>, {}{}
{}

Learn more about Model Benchmark, implementation details, and how to use the charts in our <a href="{}" target="_blank">Technical Report</a>.
"""
