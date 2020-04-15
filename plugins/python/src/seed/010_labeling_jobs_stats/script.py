import os
import supervisely_lib as sly

WORKSPACE_ID = int('%%WORKSPACE_ID%%')

#@TODO: for local integration
src_project_name = '%%IN_PROJECT_NAME%%'

# for debug
#WORKSPACE_ID = 1


api = sly.Api(server_address=os.environ['SERVER_ADDRESS'], token=os.environ['API_TOKEN'])

#### End settings. ####

workspace = api.workspace.get_info_by_id(WORKSPACE_ID)
team = api.team.get_info_by_id(workspace.team_id)

jobs = api.labeling_job.get_list(team.id)

if len(jobs) == 0:
    raise RuntimeError("Aborted: there are no labeling jobs in a team {!r}".format(team.name))

stats = [api.labeling_job.get_stats(job.id) for job in jobs]


df_jobs = sly.lj.jobs_stats(os.environ['SERVER_ADDRESS'], jobs, stats)

df_jobs_summary = sly.lj.jobs_summary(jobs)
df_images_summary = sly.lj.images_summary(jobs)
df_classes_summary = sly.lj.classes_summary(stats)
df_tags_summary = sly.lj.tags_summary(stats)


widgets = [api.report.create_table(df_jobs_summary, "Jobs summary", "How many jobs of different statuses"),
           api.report.create_table(df_images_summary, "Images summary", "How many images of different statuses"),
           api.report.create_table(df_classes_summary, "Classes summary", "How many images contain each class / how many objects of each class have been created"),
           api.report.create_table(df_tags_summary, "Tags summary", "How many images have tag / how many objects have tag"),
           api.report.create_table(df_jobs, "Labeling Jobs", "Extended statistics for every job in the team")]

report_id = api.report.create(team.id, widgets)
#print("http://192.168.1.42/reports/{}".format(report_id))

sly.logger.info('REPORT_CREATED', extra={'event_type': sly.EventType.REPORT_CREATED, 'report_id': report_id})