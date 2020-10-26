import os

from csdmd.process import run

def parse_videos(videos_folder, dest_folder): # absolute path to videos folder and destination to put sampled pictures
    video_files = os.listdir(videos_folder)

    if not os.path.isdir(os.path.join(dest_folder, 'train')):
        os.mkdir(os.path.join(dest_folder, 'train'))

    train_path = os.path.join(dest_folder, 'train')

    try:
        i = video_files.index('.DS_Store')
        video_files.pop(i)
    except:
        pass

    folder_name = 'subject'
    num_subjects = len(video_files)

    for n in range(num_subjects):
        if os.path.isdir(train_path + os.sep + folder_name + str(n)):
            continue
        os.mkdir(train_path + os.sep + folder_name + str(n))
        dest = train_path + os.sep + folder_name + str(n)
        src = videos_folder + os.sep + video_files[n]
        run(src=src, dest=dest)

    files_parsed = os.listdir(train_path)
    try:
        i = files_parsed.index('.DS_Store')
        files_parsed.pop(i)
    except:
        pass

    if num_subjects == len(files_parsed):
        print("Parsing and DMD sampling of videos successful.")

    return train_path