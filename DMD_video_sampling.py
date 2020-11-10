import os

from csdmd.process import run

def parse_videos(videos_folder, dest_folder, train_folder: bool): # absolute path to videos folder and destination to put sampled pictures
    video_files = os.listdir(videos_folder)

    if train_folder:
        os.mkdir(os.path.join(dest_folder, 'train'))
        videos_path = os.path.join(dest_folder, 'train')
    else:
        videos_path = dest_folder

    try:
        i = video_files.index('.DS_Store')
        video_files.pop(i)
    except:
        pass

    folder_name = 'subject'
    num_subjects = len(video_files)

    for n in range(num_subjects):
        if os.path.isdir(videos_path + os.sep + folder_name + str(n)):
            continue
        if train_folder:
            os.mkdir(videos_path + os.sep + folder_name + str(n))
            dest = videos_path + os.sep + folder_name + str(n)
        else:
            os.mkdir(videos_path + os.sep + folder_name + '?')
            dest = videos_path + os.sep + folder_name + '?'
        src = videos_folder + os.sep + video_files[n]
        run(src=src, dest=dest)

    files_parsed = os.listdir(videos_path)
    try:
        i = files_parsed.index('.DS_Store')
        files_parsed.pop(i)
    except:
        pass

    if num_subjects == len(files_parsed):
        print("Parsing and DMD sampling of videos successful.")

    return videos_path