# quetzal/quetzal_app/page/page_video_comparison.py
import streamlit as st
import os
from streamlit import session_state as ss

from streamlit_elements import elements, mui
from streamlit_label_kit import detection, segmentation
from glob import glob
from quetzal_app.elements.mui_components import MuiToggleButton

from quetzal.dtos.video import QueryVideo, DatabaseVideo
from quetzal.align_frames import QueryIdx, DatabaseIdx, Match
from quetzal_app.utils.utils import format_time, get_base64
from quetzal_app.page.page_state import PageState, Page
from quetzal_app.page.video_comparison_controller import (
    Controller,
    PlaybackController,
    ObjectDetectController,
    ObjectAnnotationController,
    PLAY_IDX_KEY,
)

import time

import skimage.transform 

from pathlib import Path
import base64
import pickle
import numpy as np
import uuid

from quetzal_app.elements.image_frame_component import image_frame

PAGE_NAME = "video_comparison"

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("../quetzal_logo_trans.png"))
LOGO_FILE = f"data:image/jpeg;base64,{get_base64(LOGO_FILE)}"
TITLE = "Quetzal"

BORDER_RADIUS = "0.8rem"
FRAME_IDX_TXT = "Frame Index: {}/{}"
PLAYBACK_TIME_TXT = "Playback Time: {}/{}"

WIDTH_ANNOTATE_DISPLAY = 1024
HEIGHT_ANNOTATE_DISPLAY = 576
LINE_WIDTH_ANNOTATE_DISPLAY = 2

WIDTH_SEGMENT_DEFAULT = 512
HEIGHT_SEGMENT_DEFAULT = 288

DEFAULT_LABEL_LIST = ["deer", "human", "dog", "penguin", "flamingo", "teddy bear"]

controller_dict: dict[str, Controller] = {
    PlaybackController.name: PlaybackController,
    ObjectDetectController.name: ObjectDetectController,
    ObjectAnnotationController.name: ObjectAnnotationController,
}


class VideoComparisonPage(Page):
    name = PAGE_NAME

    def __init__(self, root_state: PageState, to_page: list[callable]):
        self.root_state = root_state
        self.init_page_state(root_state)
        self.to_page = to_page

    def init_page_state(self, root_state: PageState) -> PageState:
        self.page_state = PageState(
            matches=None,
            controller=PlaybackController.name,
            warp=True,
            next_frame=False,
            info_anchor=None,
            annotated_frame={
                "query": None,
                "db": None,
                "idx": -1,
            },
            is_segment=False,
        )

        self.page_state.update(
            {
                PlaybackController.name: PlaybackController.initState(root_state),
                ObjectDetectController.name: ObjectDetectController.initState(
                    root_state
                ),
                ObjectAnnotationController.name: ObjectAnnotationController.initState(
                    root_state
                ),
                PLAY_IDX_KEY: 0,
            }
        )
        return self.page_state

    def open_file_explorer(self):
        self.init_page_state(self.root_state)
        self.to_page[0]()

    def render(self):

        st.markdown(
            f"""
                <style>
                    .block-container {{ /* Removes streamlit default white spaces in the main window*/
                            padding-top: 1.5rem;
                            padding-bottom: 1rem;
                            padding-left: 3rem;
                            padding-right: 3rem;
                        }}
                        .stSlider {{ /* Removes white spaces in the main window*/
                            padding-left: 1rem;
                            padding-right: 1rem;
                        }}
                        [class^="st-emotion-cache-"] {{ /* Removes streamlit default gap between containers, elements*/
                            gap: 0rem;
                            border-radius: {BORDER_RADIUS};
                            background-color: transparent
                        }}
                </style>
                """,
            unsafe_allow_html=True,
        )

        # Initialize Variable
        if self.page_state.matches is None:
            self.page_state.update(self.root_state.comparison_matches)            

        if "first_load" not in ss:
            ss.first_load = True

        TitleContent(
            page_state=self.page_state, to_file_explorer=self.open_file_explorer
        ).render()

        controller: dict[str, Controller] = {
            k: v(self.page_state) for k, v in controller_dict.items()
        }
        with st.container(border=True):
            FrameDisplay(self.page_state).render()

            ControllerOptions(self.page_state, self.root_state.torch_device).render()
            with st.container(border=True):
                controller['playback'].render()
                if(self.page_state.controller != 'playback'):
                    controller[self.page_state.controller].render()

            ## !! no other render beyond this!! for player-control

        if ss.first_load:
            ss.first_load = False
            st.rerun()


class TitleContent:

    def __init__(self, page_state, to_file_explorer):
        self.page_state = page_state
        self.to_file_explorer = to_file_explorer

    def handle_popover_open(self, event):
        self.page_state.info_anchor = True

    def handle_popover_close(self):
        self.page_state.info_anchor = None

    def title(self):
        with mui.Stack(
            spacing=0.5,
            direction="row",
            alignItems="center",
            justifyContent="start",
            sx={"hegiht": 51}
        ):
            mui.Avatar(
                alt="Quetzal",
                src=LOGO_FILE,
                sx={"width": 48, "height": 48 }
            )
            mui.Typography(
                TITLE,
                sx={
                    "fontSize": "h4.fontSize",
                    # /* top | left and right | bottom */
                    "margin": "0.5rem 1rem 0.25rem",
                },
            )

    def render(self):
        with elements("title"):
            with mui.Grid(container=True):
                with mui.Grid(item=True, xs=4):
                    self.title()

                with mui.Grid(item=True, xs=4):
                    mui.Box(
                        mui.icon.Compare(
                            onMouseEnter=self.handle_popover_open,
                            onMouseLeave=self.handle_popover_close,
                        ),
                        display="flex",
                        justifyContent="center",
                        alignItems="end",
                        height="100%",
                    )

                with mui.Grid(item=True, xs=4):
                    mui.Box(
                        mui.Button(
                            "Back to Home",
                            variant="text",
                            startIcon=mui.icon.Home(),
                            sx={
                                "height": "min-content",
                                "padding": 0,
                            },
                            onClick=self.to_file_explorer,
                        ),
                        display="flex",
                        justifyContent="end",
                        alignItems="end",
                        height="100%",
                    )

            with mui.Popover(
                id="mouse-over-popover",
                sx={"pointerEvents": "none"},
                open=self.page_state.info_anchor,
                anchorPosition={"top": 0, "left": 0},
                anchorOrigin={"vertical": "bottom", "horizontal": "center"},
                transformOrigin={"vertical": "bottom", "horizontal": "center"},
                onClose=self.handle_popover_close,
                disableRestoreFocus=True,
            ):
                mui.Typography(
                    'Use the slider on the "Aligned Data Frame" to compare the matched images',
                    sx={"p": 1},
                )


class FrameDisplay:

    def __init__(self, page_state):
        self.page_state = page_state
        if "is_segment" not in st.session_state:
            st.session_state.is_segment = False
        if "new_segment" not in st.session_state:
            st.session_state.new_segment = False
        if "new_detection" not in st.session_state:
            st.session_state.new_detection = False
        if "edit_query" not in st.session_state:
            st.session_state.edit_query = False
        if "edit_db" not in st.session_state:
            st.session_state.edit_db = False
        if "label_list" not in st.session_state:
            st.session_state.label_list = DEFAULT_LABEL_LIST

    def display_frame(self, labels, images, frame_lens, idxs, fps, 
                      bboxes_query=[], labels_query=[], bboxes_db=[], labels_db=[], 
                      mask_query=[], mask_db=[]):
        match self.page_state.controller:
            case ObjectAnnotationController.name:
                if st.session_state.is_segment:
                    self.display_segmentation_frame(mask_query, mask_db)
                else:
                    self.display_detection_frame(bboxes_query, labels_query, bboxes_db, labels_db)
                    
            case _:
                total_times, curr_times = [], []
                for i in range(len(frame_lens)):
                    curr_total_time, curr_show_hours = format_time(
                        frame_lens[i] / fps[i], show_hours=False, final_time=True
                    )
                    curr_time, _ = format_time(idxs[i] / fps[i], curr_show_hours)

                    total_times.append(curr_total_time)
                    curr_times.append(curr_time)

                captions = []
                for j in range(len(idxs)):
                    captions.append([FRAME_IDX_TXT.format(idxs[j], frame_lens[j]),
                                    PLAYBACK_TIME_TXT.format(curr_times[j], total_times[j])])
                    
                image_frame(
                    image_urls=images,
                    captions= captions,
                    labels=labels,
                    starting_point=0,
                    dark_mode=False,
                    key="image_comparison" + str(fps[0]) + str(fps[1])
                )

    @st.fragment
    def display_detection_frame(self, bboxes_query, labels_query, bboxes_db, labels_db):

        if "result" not in st.session_state:
            st.session_state.result = []
                
        if "result_query_out" not in st.session_state or st.session_state.new_detection:
            st.session_state.result_query_out = {"key": 0, "bbox": []}
            
        if "result_db_out" not in st.session_state or st.session_state.new_detection:
            st.session_state.result_db_out = {"key": 0, "bbox": []}
        
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        query_img = self.page_state.warp_query_frames[query_idx] if self.page_state.warp else self.page_state.query_frames[query_idx]
        database_img = self.page_state.db_frames[db_idx] if self.page_state.warp else self.page_state.db_frames[db_idx]
        
        bboxes = list(bboxes_query)+list(bboxes_db)

        label_list = st.session_state.label_list

        if st.session_state.new_detection:
            bbox_ids = [str(uuid.uuid4()) for _ in bboxes]
            label_to_idx = lambda s : label_list.index(s)
            labels = list(map(label_to_idx, list(labels_query)+list(labels_db)))
            meta_data = []
            info_dict = []
            st.session_state.result = [{"bboxes": bboxes[i], "bbox_ids": bbox_ids[i], "labels":labels[i], "label_names": label_list[labels[i]],"meta_data": meta_data, "info_dict": info_dict} for i in range(len(bboxes))]
        else:
            data = st.session_state.result
            bboxes = [item['bboxes'] for item in data]
            bbox_ids = [item['bbox_ids'] for item in data]
            meta_data = [item['meta_data'] for item in data]
            info_dict = [item['info_dict'] for item in data]
            labels = [item['labels'] for item in data]

        if label_list == []:
            print("No objects detection, loading default labels...")
            label_list = DEFAULT_LABEL_LIST

        c1, c2 = st.columns(2)
        
        with c1: 
            test_out1 = detection(
                image_path=query_img,
                bboxes=bboxes,
                bbox_ids=bbox_ids,
                bbox_format='REL_XYXY',
                labels=labels,
                # info_dict=info_dict,
                meta_data=meta_data,
                info_dict = info_dict,
                label_list=label_list,
                line_width=LINE_WIDTH_ANNOTATE_DISPLAY,
                class_select_type="radio",
                ui_position="left",
                item_editor=True,
                # item_selector=True,
                # read_only=True,
                edit_meta=True,
                bbox_show_label=True,
                key="detection_dup1",
            )

        with c2:
            test_out2 = detection(
                image_path=database_img,
                bboxes=bboxes,
                bbox_ids=bbox_ids,
                bbox_format='REL_XYXY',
                labels=labels,
                # info_dict=info_dict,
                meta_data=meta_data,
                info_dict=info_dict,
                label_list=label_list,
                line_width=LINE_WIDTH_ANNOTATE_DISPLAY,
                class_select_type="radio",
                ui_position="right",
                item_editor=True,
                # read_only=True,
                # item_selector=True,
                edit_meta=True,
                bbox_show_label=True,
                key="detection_dup2",
            )
        
        if (test_out1["key"] != st.session_state.result_query_out["key"] or test_out2["key"] != st.session_state.result_db_out["key"]) and not st.session_state.new_detection:
            if test_out1["key"] != st.session_state.result_query_out["key"]:
                st.session_state.result_query_out["key"] = test_out1["key"]
                st.session_state.result_query_out["bbox"] = test_out1["bbox"]
                
            if test_out2["key"] != st.session_state.result_db_out["key"]:
                st.session_state.result_db_out["key"] = test_out2["key"]
                st.session_state.result_db_out["bbox"] = test_out2["bbox"]

            if st.session_state.result_db_out["key"] > st.session_state.result_query_out["key"]: 
                st.session_state.result = st.session_state.result_db_out["bbox"]
            else:
                st.session_state.result = st.session_state.result_query_out["bbox"]
            try:
                st.rerun(scope="fragment")
            except st.errors.StreamlitAPIException:
                st.session_state.label_list = DEFAULT_LABEL_LIST
                st.rerun()
        
        st.session_state.new_detection = False

    def scale_masks(self, masks, width, height):
        scaled_masks = []
        for m in masks:
            scaled_masks.append(skimage.transform.resize(m, (height, width), order=0, preserve_range=True, anti_aliasing=False).tolist())
        return scaled_masks

    @st.fragment
    def display_segmentation_frame(self, mask_query, mask_db):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        query_img = self.page_state.warp_query_frames[query_idx] if self.page_state.warp else self.page_state.query_frames[query_idx]
        database_img = self.page_state.db_frames[db_idx] if self.page_state.warp else self.page_state.db_frames[db_idx]

        label_list = st.session_state.label_list
        
        if label_list == []:
            print("Err: Did not run detection or no bounding boxes detected.")

            label_list = DEFAULT_LABEL_LIST

        if "seg_result" not in st.session_state:
            st.session_state.seg_result = []

        if "seg_query_out" not in st.session_state or st.session_state.new_segment:
            st.session_state.seg_query_out = {"key": 0, "mask": []}

        if "seg_db_out" not in st.session_state or st.session_state.new_segment:
            st.session_state.seg_db_out = {"key": 0, "mask": []}
            
        masks = np.concatenate((mask_query, mask_db), axis=0).tolist()
        if st.session_state.new_segment:
            label_names = [item['label_names'] for item in st.session_state.result] if st.session_state.result else []
            meta_data = [item['meta_data'] for item in st.session_state.result] if st.session_state.result else []
            info_dict = [item['info_dict'] for item in st.session_state.result] if st.session_state.result else []
            mask_ids = [str(uuid.uuid4()) for _ in masks]
            label_to_idx = lambda s : label_list.index(s)
            labels = list(map(label_to_idx, label_names))
            st.session_state.seg_result = [{"masks": masks[i], "mask_ids": mask_ids[i], "labels":labels[i], "label_names": label_list[labels[i]], "meta_data":meta_data[i], "info_dict":info_dict[i]} for i in range(len(labels))]
        else:
            data = st.session_state.seg_result
            masks = [item['masks'] for item in data]
            mask_ids = [item['mask_ids'] for item in data]
            labels = [item['labels'] for item in data]
            meta_data = [item['meta_data'] for item in data]
            info_dict = [item['info_dict'] for item in data]
        
        c1, c2 = st.columns(2)
        masks = self.scale_masks(np.asarray(masks), WIDTH_SEGMENT_DEFAULT, HEIGHT_SEGMENT_DEFAULT)

        with c1: 
            seg_out1 = segmentation(
                image_path=query_img,
                masks=masks,
                mask_ids=mask_ids,
                labels=labels,
                label_list=label_list,
                item_editor=True,
                edit_meta=True,
                meta_data=meta_data,
                info_dict=info_dict,
                image_width=WIDTH_SEGMENT_DEFAULT,
                image_height=HEIGHT_SEGMENT_DEFAULT,
                # read_only=True,
                key="seg_dup1",
            )

        with c2:
            seg_out2 = segmentation(
                image_path=database_img,
                masks=masks,
                mask_ids=mask_ids,
                labels=labels,
                # bbox_format='XYWH',
                ui_position="right",
                # meta_data=meta_data,
                label_list=label_list,
                item_editor=True,
                item_selector=True,
                edit_meta=True,
                meta_data=meta_data,
                info_dict=info_dict,
                # read_only=True,
                image_width=WIDTH_SEGMENT_DEFAULT,
                image_height=HEIGHT_SEGMENT_DEFAULT,
                # auto_segmentation=True,
                key="seg_dup2",
            )

        if (seg_out1["key"] != st.session_state.seg_query_out["key"] or seg_out2["key"] != st.session_state.seg_db_out["key"]) and not st.session_state.new_segment:
            if seg_out1["key"] != st.session_state.seg_query_out["key"]:
                st.session_state.seg_query_out = seg_out1

            if seg_out2["key"] != st.session_state.seg_db_out["key"]:
                st.session_state.seg_db_out = seg_out2
            
            if st.session_state.seg_db_out["key"] > st.session_state.seg_query_out["key"]:
                st.session_state.seg_result = np.array(st.session_state.seg_db_out["mask"])
            else:
                st.session_state.seg_result = np.array(st.session_state.seg_query_out["mask"])
            print("Refreshing in segmentation")
            print(len(st.session_state.seg_query_out["mask"]))

            try:
                st.rerun(scope="fragment")
            except st.errors.StreamlitAPIException:
                st.session_state.label_list = DEFAULT_LABEL_LIST
                st.rerun()
        
        st.session_state.new_segment = False

        

    def render(self):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        query: QueryVideo = self.page_state.query
        database: DatabaseVideo = self.page_state.database
        bboxes_query, labels_query, bboxes_db, labels_db, mask_query, mask_db, label_list = [], [], [], [], [], [], []
        
        match self.page_state.controller:
            case PlaybackController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]
            case ObjectDetectController.name if self.page_state.annotated_frame[
                "idx"
            ] == ss.slider:
                query_img = self.page_state.annotated_frame["query"]
                database_img = self.page_state.annotated_frame["db"]
            case ObjectAnnotationController.name if self.page_state.annotated_frame[
                "idx"
            ] != -1:
                query_img = self.page_state.warp_query_frames[query_idx] if self.page_state.warp else self.page_state.query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx] if self.page_state.warp else self.page_state.db_frames[db_idx]
                labels_query = self.page_state.annotated_frame["labels_query"]
                labels_db = self.page_state.annotated_frame["labels_db"]

                if(st.session_state.new_detection):
                    st.session_state.label_list = self.page_state.annotated_frame["label_list"]
                    bboxes_query = self.page_state.annotated_frame["bboxes_query"]
                    bboxes_db = self.page_state.annotated_frame["bboxes_db"]
                    st.session_state.is_segment = False
                elif(st.session_state.new_segment):
                    mask_query = self.page_state.annotated_frame["mask_query"]
                    labels_query = self.page_state.annotated_frame["labels_query"]
                    mask_db = self.page_state.annotated_frame["mask_db"]
                    labels_db = self.page_state.annotated_frame["labels_db"]
                    st.session_state.is_segment = True

            case ObjectDetectController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]

            case ObjectAnnotationController.name if self.page_state.warp:
                query_img = self.page_state.warp_query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]
            case _:
                query_img = self.page_state.query_frames[query_idx]
                database_img = self.page_state.db_frames[db_idx]

        query_img_base64 = f"data:image/jpeg;base64,{get_base64(query_img)}"
        db_img_base64 = f"data:image/jpeg;base64,{get_base64(database_img)}"

        labels = ["Query Frame: " + query.name,"Aligned Database Frame: " + database.name]
        images = [[query_img_base64], [query_img_base64, db_img_base64]]
        frame_lens = [len(self.page_state.query_frames), len(self.page_state.db_frames)]
        idxs = [query_idx, db_idx]
        fps = [QueryVideo.FPS, DatabaseVideo.FPS]
    
        self.display_frame(
            labels=labels, 
            images=images, 
            frame_lens=frame_lens, 
            idxs=idxs, 
            fps=fps,
            bboxes_query=bboxes_query,
            labels_query=labels_query,
            bboxes_db=bboxes_db,
            labels_db=labels_db, 
            mask_query=mask_query,
            mask_db=mask_db,
            )
            
class ControllerOptions:

    toggle_buttons_style = {
        "gap": "1rem",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "bgcolor": "grey.200",
            "gap": "0.5rem",
            "py": "0.2rem",
            "&:not(:last-of-type)": {
                "borderRadius": "0.5rem",
            },
            "&:not(:first-of-type)": {
                "borderRadius": "0.5rem",
            },
        },
        "& .MuiToggleButton-root": {
            "&.Mui-selected": {
                "color": "white",
                "bgcolor": "black",
                "&:hover": {"bgcolor": "black"},
            },
            "&:hover": {"bgcolor": "grey.300"},
        },
    }

    def __init__(self, page_state, torch_device):
        self.page_state = page_state
        self.torch_device = torch_device

    def handleSwitch(self):
        self.page_state.warp = not self.page_state.warp

    def stotrePageState(self, page):
        controller_dict[page].storeState(self.page_state[page])

    def loadPageState(self, page):
        controller_dict[page].loadState(self.page_state[page])

    def handleController(self, event, controller):
        if controller is not None:
            self.stotrePageState(self.page_state.controller)
            self.loadPageState(controller)
            self.page_state.controller = controller

    def render(self):
        toggle_buttons = [
            MuiToggleButton(PlaybackController.name, "PlayArrow", "Playback Control"),
            MuiToggleButton(
                ObjectDetectController.name, "CenterFocusStrong", "Object Detection"
            ),
            MuiToggleButton(
                ObjectAnnotationController.name, "Create", "Object Annotation"
            ),
        ]

        with elements("tabs"):
            with mui.Stack(
                spacing=2,
                direction="row",
                alignItems="start",
                justifyContent="space-between",
                sx={"my": 0, "maxHeight": "calc(30.39px - 22.31px + 1rem)"},
            ):
                with mui.ToggleButtonGroup(
                    value=self.page_state.controller,
                    onChange=self.handleController,
                    exclusive=True,
                    sx=self.toggle_buttons_style,
                ):
                    for button in toggle_buttons:
                        button.render()

                with mui.Stack(
                    direction="row",
                    spacing=0,
                    alignItems="start",
                    sx={"maxHeight": "calc(30.39px - 22.31px + 1rem)"},
                ):
                    mui.Typography("Image Warp", sx={"ml": "0.3rem", "py": "7px"})
                    mui.Switch(checked=self.page_state.warp, onChange=self.handleSwitch)