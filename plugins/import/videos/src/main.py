# coding: utf-8

import json
import os.path
import skvideo.io
import supervisely_lib as sly

from PIL import Image as pil_image

import dhash

DEFAULT_STEP = 25
STEP = 'step'
START_FRAME = 'start_frame'
END_FRAME = 'end_frame'
SKIP_FRAMES = 'skip_frames'
DHASH_MIN_HAMMING_DISTANCE = 'dhash_min_hamming_distance'


def convert_video():
    task_settings = json.load(open(sly.TaskPaths.TASK_CONFIG_PATH, 'r'))

    convert_options = task_settings['options']

    step = convert_options.get(STEP)
    if step is not None:
        step = int(step)
    else:
        sly.logger.warning('step parameter not found. set to default: {}'.format(DEFAULT_STEP))
        step = DEFAULT_STEP

    start_frame = convert_options.get(START_FRAME, 0)
    end_frame = convert_options.get(END_FRAME, float('Inf'))
    skip_frames = set(convert_options.get(SKIP_FRAMES, []))
    dhash_min_hamming_distance = convert_options.get(DHASH_MIN_HAMMING_DISTANCE, 0)

    paths = sly.fs.list_files(sly.TaskPaths.DATA_DIR)
    video_paths = []
    for path in paths:
        if sly.video.has_valid_ext(path):
            video_paths.append(path)
        else:
            sly.logger.warning("Video file '{}' has unsupported extension. Skipped. Supported extensions: {}"
                               .format(path, sly.video.ALLOWED_VIDEO_EXTENSIONS))

    if len(video_paths) == 0:
        raise RuntimeError("Videos not found!")

    project_dir = os.path.join(sly.TaskPaths.RESULTS_DIR, task_settings['res_names']['project'])
    project = sly.Project(directory=project_dir, mode=sly.OpenMode.CREATE)
    for video_path in video_paths:
        try:
            video_relpath = os.path.relpath(video_path, sly.TaskPaths.DATA_DIR)
            ds_name = video_relpath.replace('/', '__')
            ds = project.create_dataset(ds_name=ds_name)

            vreader = skvideo.io.FFmpegReader(video_path)

            vlength = vreader.getShape()[0]
            progress = sly.Progress('Import video: {}'.format(ds_name), vlength)

            prev_dhash = None

            for frame_id, image in enumerate(vreader.nextFrame()):
                if ((start_frame <= frame_id) and (frame_id <= end_frame or end_frame < 0)
                        and ((frame_id - start_frame) % step == 0)
                        and frame_id not in skip_frames):

                    # Only keep track of dhash values if the filtering is actually enabled.
                    if dhash_min_hamming_distance > 0:
                        curr_dhash = dhash.dhash_int(pil_image.fromarray(image))
                        dhash_distance = None
                        if prev_dhash is not None:
                            dhash_distance = dhash.get_num_bits_different(curr_dhash, prev_dhash)
                            sly.logger.info('Frame {}. dHash Hamming distance to previously imported frame: {}'.format(
                                frame_id, dhash_distance))

                        if dhash_distance is not None and dhash_distance < dhash_min_hamming_distance:
                            continue
                        # Only update the prev_dhash value if we are going to actually import this frame.
                        prev_dhash = curr_dhash

                    img_name = "frame_{:05d}".format(frame_id)
                    ds.add_item_np(img_name + '.png', image)
                progress.iter_done_report()

        except Exception as e:
            exc_str = str(e)
            sly.logger.warn('Input video skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                'exc_str': exc_str,
                'video_file': video_path,
            })


def main():
    convert_video()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('VIDEO_ONLY_IMPORT', main)