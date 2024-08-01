from quetzal_app.page.page_video_comparison import VideoComparisonPage
from quetzal_app.page.page_video_comparison_real_time import VideoComparisonRealTimePage
from quetzal_app.page.page_video_comparison_stream import VideoComparisonStreamPage

from quetzal_app.page.page_file_explorer import FileExplorerPage
from quetzal_app.page.page_state import AppState, PageState, Page

from quetzal.dtos.video import QueryVideo, DatabaseVideo, QuetzalFile
import pickle


import streamlit as st
from streamlit import session_state as ss
import argparse
import torch

from threading import Lock
import os
from pathlib import Path

import streamlit_elements
import re

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("quetzal_logo_trans.png"))


dataset_layout_help = """
    Dataset structure:
    root_datasets_dir/
    |
    ├──user_name1/
    |   |
    |   ├── project_name1/
    |   |   ├── video_name1.mp4
    |   |   ├── video_name2.mp4
    |   |   ├── ...
    |   |   |
    |   |   ├── subproject_name/
    |   |   |   ├──video_name1.mp4
    |   |   |   └── ...
    |   |   └── ...
    |   |
    |   |
    |   └── project_name2/
    |
    ├──user_name2/
    └── ...

    metadata_directory/

    ├── user_name1.info.txt
    ├── user_name1.meta.txt
    ├── user_name1/
    |   |
    |   ├── project_name1.info.txt
    |   ├── project_name1.meta.txt
    |   ├── project_name1/
    |   |   ├── video_name1.mp4.info.txt
    |   |   ├── video_name1.mp4.meta.txt
    |   |   |
    |   |   ├── video_name2.mp4.info.txt
    |   |   ├── video_name2.mp4.meta.txt
    |   |   |
    |   |   ├── ...
    |   |   |
    |   |   ├── database/
    |   |   |   ├── video_name1/
    |   |   |   |   ├── frames_{fps}_{resolution}/
    |   |   |   |   |   ├── frame_%05d.jpg
    |   |   |   |   |   └── ...
    |   |   |   |   └── ...
    |   |   |   └── video_name2/
    |   |   |       ├── frames_{fps}_{resolution}/
    |   |   |       |   ├── frame_%05d.jpg
    |   |   |       |   └── ...
    |   |   |       └── ...
    |   |   |
    |   |   ├── query/
    |   |   |   ├── video_name2/
    |   |   |   |   ├── frames_{fps}_{resolution}/
    |   |   |   |   |   ├── frame_%05d.jpg
    |   |   |   |   |   └── ...
    |   |   |   |   └── ...
    |   |   |   └── ...
    |   |   |
    |   |   ├── subproject_name.info.txt
    |   |   ├── subproject_name.meta.txt
    |   |   ├── subproject_name/
    |   |   |   ├──video_name1.mp4.info.txt
    |   |   |   ├──video_name1.mp4.meta.txt
    |   |   |   └── ...
    |   |   └── ...
    |   |
    |   |
    |   └── project_name2/
    |
    ├── user_name1.info.txt
    ├── user_name1.meta.txt
    ├── user_name2/
    └── ...
        """

st.set_page_config(layout="wide", page_title="Quetzal", page_icon=LOGO_FILE)

@st.cache_data
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="This program aligns Two Videos",
        epilog=dataset_layout_help,
    )

    parser.add_argument(
        "-d",
        "--dataset-root",
        default="./data/home/root",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "-m",
        "--metadata-root",
        default="./data/meta_data/root",
        help="Meta data directory of datasets",
    )
    parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=True)
    parser.add_argument("--cuda-device", help="Select cuda device", default=0, type=int)
    parser.add_argument("-u", "--user", default="default_user")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    meta_data_root = args.metadata_root

    available_gpus = torch.cuda.device_count()
    print(f"Avaliable GPU={available_gpus}")
    if args.cuda and available_gpus > 0:
        cuda_device = args.cuda_device if args.cuda_device < available_gpus else 0
        torch_device = torch.device("cuda:" + str(cuda_device))
    else:
        torch_device = torch.device("cpu")

    print(torch_device)

    return dataset_root, meta_data_root, cuda_device, torch_device, args.user


dataset_root, meta_data_root, cuda_device, torch_device, user = parse_args()
headers = st.context.headers
# user = headers.get("X-Forwarded-User", user)
user = user

page_list: list[Page] = [
    FileExplorerPage,
    VideoComparisonPage, 
    VideoComparisonRealTimePage,
    VideoComparisonStreamPage
]

page_dict: dict[str, Page] = {page.name: page for page in page_list}

if "page_states" not in ss:
    app_state = AppState()

    # for development
    with open('query.pkl', 'rb') as f:
        q = pickle.load(f)
        query_video = QueryVideo.from_quetzal_file(QuetzalFile(path=q, root_dir=dataset_root, metadata_dir=meta_data_root, user=user))
        f.close()
    with open('db.pkl', 'rb') as f:
        db = pickle.load(f)
        database_video = DatabaseVideo.from_quetzal_file(QuetzalFile(path=db, root_dir=dataset_root, metadata_dir=meta_data_root, user=user))
        f.close()
    with open('matches.pkl', 'rb') as f:
        matches = pickle.load(f)
        f.close()
    with open('warp_query_frame_list.pkl', 'rb') as f:
        warp_query_frame_list = pickle.load(f)
        f.close()
    with open('query_frame_list.pkl', 'rb') as f:
        query_frame_list = pickle.load(f)
        f.close()
    with open('db_frame_list.pkl', 'rb') as f:
        db_frame_list = pickle.load(f)
        f.close()
    
    root_state = PageState(
        root_dir=dataset_root,
        metadata_dir=meta_data_root,
        cuda_device=cuda_device,
        torch_device=torch_device,
        page=FileExplorerPage.name,
        user=user,
        comparison_matches=None,
    )

    root_state["comparison_matches"] = {
        "query": query_video,
        "database": database_video,
        "matches": matches,
        "query_frames": query_frame_list,
        "db_frames": db_frame_list,
        "warp_query_frames": warp_query_frame_list,
    }

    # root_state.page = FileExplorerPage.name
    # root_state.page = VideoComparisonStreamPage.name
    root_state.page = VideoComparisonPage.name


    def build_to_page(page: Page):
        def to_page():
            root_state.page = page.name
            print("to_page", page.name)

        return to_page

    to_page = [build_to_page(page) for page in page_list]

    ss.pages = dict()

    app_state.root = root_state
    for key, page in page_dict.items():
        page_object = page(root_state=root_state, to_page=to_page)
        ss.pages[key] = page_object
        app_state[key] = page_object.page_state

    ss.page_states = app_state
    ss.lock = Lock()


# script to fix compatibility issues for streamlit_elements with newer streamlit versions
# **Note, will have to restart application if a broken import is fixed
def patch_streamlit_elements():

    # issue: https://github.com/okld/streamlit-elements/issues/35
    relative_file_path = 'core/callback.py'
    library_root = list(streamlit_elements.__path__)[0]
    file_path = os.path.join(library_root, relative_file_path)

    # Read broken file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    broken_import = 'from streamlit.components.v1 import components'
    fixed_import = 'from streamlit.components.v1 import custom_component as components\n'

    # Fix broken import line
    for index, line in enumerate(lines):

        if re.match(broken_import, line):
            print(f'Replaced broken import in {file_path}, please restart application.')
            lines[index] = fixed_import

    # Update broken file with fix
    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
    patch_streamlit_elements()
# import pickle
# from quetzal.dtos.video import DatabaseVideo, QueryVideo
# with open("../test_matches.pkl", 'rb') as f:
#     comparison_matches = pickle.load(f)

# database_video = DatabaseVideo(
#     path = comparison_matches["database"],
#     root_dir=dataset_root,
#     metadata_dir=meta_data_root,
#     user=user,
# )

# query_video = QueryVideo(
#     path = comparison_matches["query"],
#     root_dir=dataset_root,
#     metadata_dir=meta_data_root,
#     user=user,
# )

# comparison_matches["database"] = database_video
# comparison_matches["query"] = query_video

# ss.page_states.root["comparison_matches"] = comparison_matches



ss.pages[ss.page_states.root.page].render()
