from abc import ABC, abstractmethod
import datetime
from pathlib import Path
import time
from typing import NewType

import streamlit as st
from streamlit import session_state as ss
from streamlit_elements import elements, mui
from streamlit_extras.stylable_container import stylable_container
from streamlit_tags import st_tags
import json
import base64

from quetzal.align_frames import DatabaseIdx, Match, QueryIdx
from quetzal.engines.detection_engine.grounding_sam_engine import GroundingSAMEngine
from quetzal.engines.engine import ObjectDetectionEngine


DEFAULT_BOX_TH = 0.25
DEFAULT_TEXT_TH = 0.25
DEFAULT_SLIDER_VAL = 0
DEFAULT_OBJECT_PROMPT = ["objects"]

SLIDER_KEY = "slider"
PLAYBACK_KEY = "run_playback"
PLAY_IDX_KEY = "playback_idx"
WAKEUP_TIME = "wakeup_time"
SLIDER_BOX_TH_KEY = "slider_box_th"
SLIDER_TXT_TH_KEY = "slider_text_th"
CLASS_PROMPT_KEY = "class_prompts"
DETECTOR_KEY = "object_detector"
DETECTOR_NAME_KEY = "object_detector_name"
SELECT_MODEL_KEY = "object_detector_choice"

QUERY_ANNOTATE_IMG = str(
    Path(__file__).parent.joinpath("../tmp/quetzal_annotated_query.jpg")
)
DB_ANNOTATE_IMG = str(Path(__file__).parent.joinpath("../tmp/quetzal_annotated_db.jpg"))

ANNOTATE_SAVE_PATH = str(
    Path(__file__).parent.joinpath("../annotations")
)

## List of Object Detector to Use
DetectorName = NewType("DetectorName", str)

detector_list = [GroundingSAMEngine]
detector_dict: dict[DetectorName, ObjectDetectionEngine] = {
    model.name: model for model in detector_list
}

@st.cache_resource
def getDetectionEngine(detector: DetectorName, torch):
    return detector_dict[detector](device=torch)


class Controller(ABC):
    name = "default"

    @staticmethod
    @abstractmethod
    def initState(root_state) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def storeState(page_state):
        pass

    @staticmethod
    @abstractmethod
    def loadState(page_state):
        pass

    @abstractmethod
    def render(self):
        pass


class PlaybackController(Controller):
    name = "playback"

    PADDING = 8
    GAP = 8
    ICON_SIZE = 48
    BUTTON_SIZE = ICON_SIZE + 2 * PADDING
    PLAYER_CONTROLER_W = BUTTON_SIZE * 5 + GAP * (5 - 1) + 2 * PADDING

    @staticmethod
    def initState(root_state) -> dict:
        return {SLIDER_KEY: DEFAULT_SLIDER_VAL}

    @staticmethod
    def storeState(page_state):
        page_state[SLIDER_KEY] = ss[SLIDER_KEY]

    @staticmethod
    def loadState(page_state):
        ss[SLIDER_KEY] = page_state[SLIDER_KEY]

    def initPlayBackController(self):
        if PLAYBACK_KEY not in ss:
            ss[PLAYBACK_KEY] = False
            ss.next_frame = False

        if SLIDER_KEY not in ss:
            ss[SLIDER_KEY] = self.page_state[self.name][SLIDER_KEY]
            ss[SLIDER_KEY + "value"] = ss[SLIDER_KEY]

        if ss[PLAYBACK_KEY] and ss.next_frame:
            self.change_slider(1)

    def __init__(self, page_state):
        self.page_state = page_state
        self.slider_min = 0
        self.slider_max = len(page_state.matches) - 1
        self.initPlayBackController()

    def set_slider(self, val):
        ss[SLIDER_KEY] = val

        if val < self.slider_min:
            ss[SLIDER_KEY] = self.slider_min

        elif val >= self.slider_max:
            ss[SLIDER_KEY] = self.slider_max
            if ss[PLAYBACK_KEY]:
                self.toggle_play()

        else:
            ss[SLIDER_KEY] = val

        ss[SLIDER_KEY + "value"] = ss[SLIDER_KEY]

    def change_slider(self, val=0):
        self.set_slider(ss[SLIDER_KEY] + val)
        self.page_state[PLAY_IDX_KEY] = ss[SLIDER_KEY]
        self.page_state[PlaybackController.name][SLIDER_KEY] = ss[SLIDER_KEY]

    def toggle_play(self):
        ss[PLAYBACK_KEY] = not ss[PLAYBACK_KEY]
        if ss[PLAYBACK_KEY]:
            ss[WAKEUP_TIME] = datetime.datetime.now() + datetime.timedelta(seconds=0.5)

    def render_player(self):
        PLAYER_CONTROLER_W = self.PLAYER_CONTROLER_W
        BUTTON_SIZE = self.BUTTON_SIZE
        PADDING = self.PADDING
        GAP = self.GAP

        with stylable_container(
            key="playback_controller",
            css_styles=f"""{{
                    display: block;
                    & div {{
                        min-width: {PLAYER_CONTROLER_W}px;
                    }}
                    & iframe {{
                        min-width: {PLAYER_CONTROLER_W}px;
                    }}
                }}
                """,
        ):
            with elements("playback_controller_element"):
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    sx={
                        "my": 0,
                        "maxHeight": f"calc({BUTTON_SIZE}px - 22.31px + {PADDING}px)",
                        "minWidth": BUTTON_SIZE * 5 + GAP * (5 - 1),
                    },
                    alignItems="start",
                    justifyContent="center",
                ):
                    with mui.IconButton(
                        onClick=lambda: self.change_slider(-5), sx={"fontSize": 48}
                    ):
                        mui.icon.KeyboardDoubleArrowLeft(fontSize="inherit")

                    mui.IconButton(
                        children=mui.icon.KeyboardArrowLeft(fontSize="inherit"),
                        onClick=lambda: self.change_slider(-1),
                        sx={"fontSize": 48},
                    )

                    with mui.IconButton(onClick=self.toggle_play, sx={"fontSize": 48}):
                        if ss[PLAYBACK_KEY]:
                            mui.icon.Pause(fontSize="inherit")
                        else:
                            mui.icon.PlayArrow(fontSize="inherit")

                    mui.IconButton(
                        children=mui.icon.KeyboardArrowRight(fontSize="inherit"),
                        onClick=lambda: self.change_slider(1),
                        sx={"fontSize": 48},
                    )

                    mui.IconButton(
                        children=mui.icon.KeyboardDoubleArrowRight(fontSize="inherit"),
                        onClick=lambda: self.change_slider(5),
                        sx={"fontSize": 48},
                    )

    def render_slider(self):
        PLAYER_CONTROLER_W = self.PLAYER_CONTROLER_W

        with stylable_container(
            key=SLIDER_KEY,
            css_styles=f"""{{
                    display: block;
                    .stSlider {{
                        position: absolute;
                        right: calc(45px - 2%) !important;
                        width: calc(102% - 45px) !important;
                        max-width: calc(152% - 45px - {PLAYER_CONTROLER_W}px) !important;   
                        padding-right: 0.5rem;
                    }}
                    
                }}
                """,
        ):
            st.slider(
                value=self.page_state[PlaybackController.name][SLIDER_KEY],
                label="Query Frame Index",
                min_value=self.slider_min,
                max_value=self.slider_max,
                step=1,
                format=f"%d",
                key=SLIDER_KEY,
                on_change=self.change_slider,
            )

    def render_slider_value(self):
        with stylable_container(
            key=SLIDER_KEY + "value",
            css_styles=f"""{{
                    display: block;
                    .stNumberInput {{
                        position: absolute;
                        right: 0px !important;
                        width: 45px !important;
                    }}
                }}
                """,
        ):
            st.number_input(
                value=self.page_state[PlaybackController.name][SLIDER_KEY],
                label=" ",
                key=SLIDER_KEY + "value",
                step=1,
                min_value=self.slider_min,
                max_value=self.slider_max,
                on_change=lambda: self.set_slider(ss[SLIDER_KEY + "value"]),
            )

    def render(self):
        cc1, cc2, cc3 = st.columns([25, 50, 1])
        with cc1:
            self.render_player()
        with cc2:
            self.render_slider()
        with cc3:
            self.render_slider_value()

        if ss[PLAYBACK_KEY]:
            ss.next_frame = False
            curr_wakeup_time = ss[WAKEUP_TIME]

            sleep_duration = max(
                0, (curr_wakeup_time - datetime.datetime.now()).total_seconds()
            )
            time.sleep(sleep_duration)

            if ss[WAKEUP_TIME] == curr_wakeup_time:  ## no other instance modified it
                ss[WAKEUP_TIME] += datetime.timedelta(seconds=0.5)
                ss.next_frame = True
                # self.change_slider(1)
                st.rerun()


class ObjectDetectController(Controller):
    name = "object_detection"
    select_dict = {model.name: num for num, model in enumerate(detector_list)}

    @staticmethod
    def initState(root_state) -> dict:
        values = {
            SLIDER_BOX_TH_KEY: DEFAULT_BOX_TH,
            SLIDER_TXT_TH_KEY: DEFAULT_TEXT_TH,
            CLASS_PROMPT_KEY: DEFAULT_OBJECT_PROMPT,
            "torch_device": root_state.torch_device,
            DETECTOR_KEY: None,
            DETECTOR_NAME_KEY: GroundingSAMEngine.name,
        }
        return values

    @staticmethod
    def storeState(page_state):
        page_state[SLIDER_BOX_TH_KEY] = ss[SLIDER_BOX_TH_KEY]
        page_state[SLIDER_TXT_TH_KEY] = ss[SLIDER_TXT_TH_KEY]
        page_state[CLASS_PROMPT_KEY] = ss[CLASS_PROMPT_KEY]

    @staticmethod
    def loadState(page_state):
        ss[SLIDER_BOX_TH_KEY] = page_state[SLIDER_BOX_TH_KEY]
        ss[SLIDER_TXT_TH_KEY] = page_state[SLIDER_TXT_TH_KEY]
        ss[CLASS_PROMPT_KEY] = page_state[CLASS_PROMPT_KEY]
        if page_state[DETECTOR_KEY] is None:
            page_state[DETECTOR_KEY] = getDetectionEngine(
                page_state[DETECTOR_NAME_KEY], page_state["torch_device"]
            )
        elif page_state[DETECTOR_KEY].name != page_state[DETECTOR_NAME_KEY]:
            page_state[DETECTOR_KEY] = getDetectionEngine(
                page_state[DETECTOR_NAME_KEY], page_state["torch_device"]
            )

    def initObjectDetectController(self):
        if CLASS_PROMPT_KEY not in ss:
            ss[CLASS_PROMPT_KEY] = DEFAULT_OBJECT_PROMPT

    def __init__(self, page_state):
        self.page_state = page_state
        self.detections_query = None
        self.labels_query = None
        self.detections_database = None
        self.labels_database = None
        # page_state[self.name][DETECTOR_NAME_KEY] = GroundingSAMEngine.name

    def _run_detection(
        self, text_prompt, input_img, output_file, box_threshold, text_threshold, isQuery
    ):
        detector: ObjectDetectionEngine = self.page_state[self.name][DETECTOR_KEY]
        if isQuery:
            _, self.detections_query, self.labels_query = detector.generate_masked_images(
                input_img, text_prompt, output_file, box_threshold, text_threshold
            )

        else:
            _, self.detections_database, self.labels_database = detector.generate_masked_images(
                input_img, text_prompt, output_file, box_threshold, text_threshold
            )

    def run_detection(self):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]

        query_img_orig = self.page_state.query_frames[query_idx]
        database_img_aligned = self.page_state.db_frames[db_idx]

        self._run_detection(
            text_prompt=ss[CLASS_PROMPT_KEY],
            input_img=query_img_orig,
            output_file=QUERY_ANNOTATE_IMG,
            box_threshold=ss[SLIDER_BOX_TH_KEY],
            text_threshold=ss[SLIDER_TXT_TH_KEY],
            isQuery=True
        )

        self._run_detection(
            text_prompt=ss[CLASS_PROMPT_KEY],
            input_img=database_img_aligned,
            output_file=DB_ANNOTATE_IMG,
            box_threshold=ss[SLIDER_BOX_TH_KEY],
            text_threshold=ss[SLIDER_TXT_TH_KEY],
            isQuery=False
        )

        self.page_state.annotated_frame = {
            "query": QUERY_ANNOTATE_IMG,
            "db": DB_ANNOTATE_IMG,
            "bboxes_query": self.detections_query,
            "labels_query": self.labels_query,
            "bboxes_db": self.detections_database,
            "labels_db": self.labels_database,
            "idx": self.page_state[PlaybackController.name][SLIDER_KEY],
        }

    def render_prompt(self):
        c1, c2 = st.columns([100, 1])

        with c1:
            with stylable_container(
                key="class_prompt_list",
                css_styles="""{
                    display: block;
                    padding: 0.5em 0.5em 0em; /*top right&left bottom*/
                    width: calc(101% - 133px);
                    }
                """,
            ):
                with st.container():
                    st_tags(
                        label=" ",
                        text="Detection Prompt: Press enter to add more class",
                        value=ss[CLASS_PROMPT_KEY],
                        suggestions=[],
                        maxtags=10,
                        key=CLASS_PROMPT_KEY,
                    )

        with c2:
            with stylable_container(
                key="object_detect_button",
                css_styles="""{
                        display: block;
                        position: absolute;
                        width: 133px !important;
                        right: 0px;
                        
                        & div {
                            width: 133px !important;
                            height: auto; 
                        }
                        
                        & iframe {
                            width: 133px !important;
                            height: 57px;
                        }                    
                    }
                    """,
            ):
                with elements("object_detection_controller"):
                    mui.Button(
                        children="Detect",
                        variant="contained",
                        startIcon=mui.icon.Search(),
                        onClick=self.run_detection,
                        size="large",
                        sx={
                            "bgcolor": "grey.800",
                            "borderRadius": "0.5rem",
                            "width": "117.14px",
                        },
                    )

    def change_model(self):
        self.page_state[self.name][DETECTOR_NAME_KEY] = ss[SELECT_MODEL_KEY]
        self.page_state[self.name][DETECTOR_KEY] = getDetectionEngine(
            self.page_state[self.name][DETECTOR_NAME_KEY],
            self.page_state[self.name]["torch_device"],
        )

    def render(self):
        self.render_prompt()
        # pass

        cc1, cc2, cc3 = st.columns([1, 3, 3])
        with cc1:
            st.selectbox(
                label="Choose Detection Model",
                options=[key for key in detector_dict.keys()],
                key=SELECT_MODEL_KEY,
                index=self.select_dict[self.page_state[self.name][DETECTOR_NAME_KEY]],
                on_change=self.change_model,
            )
        with cc2:
            st.slider(
                value=self.page_state[ObjectDetectController.name][SLIDER_BOX_TH_KEY],
                label="Box Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format=f"%0.2f",
                key=SLIDER_BOX_TH_KEY,
            )
        with cc3:
            st.slider(
                value=self.page_state[ObjectDetectController.name][SLIDER_TXT_TH_KEY],
                label="Text Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format=f"%0.2f",
                key=SLIDER_TXT_TH_KEY,
            )



class ObjectAnnotationController(Controller):
    name = "object_annotate"
    select_dict = {model.name: num for num, model in enumerate(detector_list)}
    
    @staticmethod
    def initState(root_state) -> dict:
        values = {
            SLIDER_BOX_TH_KEY: DEFAULT_BOX_TH,
            SLIDER_TXT_TH_KEY: DEFAULT_TEXT_TH,
            CLASS_PROMPT_KEY: DEFAULT_OBJECT_PROMPT,
            "torch_device": root_state.torch_device,
            DETECTOR_KEY: None,
            DETECTOR_NAME_KEY: GroundingSAMEngine.name,
        }
        return values

    @staticmethod
    def storeState(page_state):
        page_state[SLIDER_BOX_TH_KEY] = ss[SLIDER_BOX_TH_KEY]
        page_state[SLIDER_TXT_TH_KEY] = ss[SLIDER_TXT_TH_KEY]
        page_state[CLASS_PROMPT_KEY] = ss[CLASS_PROMPT_KEY]

    @staticmethod
    def loadState(page_state):
        ss[SLIDER_BOX_TH_KEY] = page_state[SLIDER_BOX_TH_KEY]
        ss[SLIDER_TXT_TH_KEY] = page_state[SLIDER_TXT_TH_KEY]
        ss[CLASS_PROMPT_KEY] = page_state[CLASS_PROMPT_KEY]
        if page_state[DETECTOR_KEY] is None:
            page_state[DETECTOR_KEY] = getDetectionEngine(
                page_state[DETECTOR_NAME_KEY], page_state["torch_device"]
            )
        elif page_state[DETECTOR_KEY].name != page_state[DETECTOR_NAME_KEY]:
            page_state[DETECTOR_KEY] = getDetectionEngine(
                page_state[DETECTOR_NAME_KEY], page_state["torch_device"]
            )

    def initObjectAnnotationController(self):
        if CLASS_PROMPT_KEY not in ss:
            ss[CLASS_PROMPT_KEY] = DEFAULT_OBJECT_PROMPT

    def __init__(self, page_state):
        self.page_state = page_state
        self.detections_query = None
        self.labels_query = None
        self.detections_database = None
        self.labels_database = None

        self.mask_query = None
        self.mask_db = None

        self.label_list = []
        
        self.is_query = lambda x : x["label_names"][-1:] == 'q' 
        self.is_db = lambda x : x["label_names"][-2:] == 'db'

        # page_state[self.name][DETECTOR_NAME_KEY] = GroundingSAMEngine.name

    def _run_detection(
        self, text_prompt, input_img, output_file, box_threshold, text_threshold, isQuery, idx
    ):
        detector: ObjectDetectionEngine = self.page_state[self.name][DETECTOR_KEY]
        if isQuery:
            _, self.detections_query, self.labels_query = detector.generate_masked_images(
                input_img, text_prompt, output_file, box_threshold, text_threshold, False
            )

        else:
            _, self.detections_database, self.labels_database, = detector.generate_masked_images(
                input_img, text_prompt, output_file, box_threshold, text_threshold, False
            )

    def run_detection(self):
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]

        idx = self.page_state[PLAY_IDX_KEY] 

        if self.page_state.warp:
            query_img_orig = self.page_state.warp_query_frames[query_idx]
        else:
            query_img_orig = self.page_state.query_frames[query_idx]
        database_img_aligned = self.page_state.db_frames[db_idx]
    
        self._run_detection(
            text_prompt=ss[CLASS_PROMPT_KEY],
            input_img=query_img_orig,
            output_file=QUERY_ANNOTATE_IMG,
            box_threshold=ss[SLIDER_BOX_TH_KEY],
            text_threshold=ss[SLIDER_TXT_TH_KEY],
            isQuery=True,
            idx=idx
        )

        self._run_detection(
            text_prompt=ss[CLASS_PROMPT_KEY],
            input_img=database_img_aligned,
            output_file=DB_ANNOTATE_IMG,
            box_threshold=ss[SLIDER_BOX_TH_KEY],
            text_threshold=ss[SLIDER_TXT_TH_KEY],
            isQuery=False,
            idx=idx
        )

        label_to_query = lambda s : s + "_q"
        label_to_db = lambda s : s +"_db"
        labels_query = list(map(label_to_query, self.labels_query))
        labels_db = list(map(label_to_db, self.labels_database))
        self.label_list = list(set(list(labels_query) + list(labels_db)))

        print("In detection: ", self.label_list)

        self.page_state.annotated_frame = {
            "query": QUERY_ANNOTATE_IMG,
            "db": DB_ANNOTATE_IMG,
            "bboxes_query": self.detections_query,
            "labels_query": labels_query,
            "bboxes_db": self.detections_database,
            "labels_db": labels_db,
            "label_list": self.label_list,
            "mask_query": [],
            "mask_db" : [],
            "idx": idx,
        }
        st.session_state.new_detection = True


    def _segment_annotation(self, input_img, output_file, xyxy, isQuery):
        detector: ObjectDetectionEngine = self.page_state[self.name][DETECTOR_KEY]
        if isQuery:
            self.annotated_image_query, self.mask_query = detector.generate_segmented_images(
                input_img, output_file, xyxy
            )

        else:
            self.annotated_image_db, self.mask_db = detector.generate_segmented_images(
                input_img, output_file, xyxy
            )

    def segment_annotation(self):
            match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
            query_idx: QueryIdx = match[0]
            db_idx: DatabaseIdx = match[1]

            if self.page_state.warp:
                query_img_orig = self.page_state.warp_query_frames[query_idx]
            else:
                query_img_orig = self.page_state.query_frames[query_idx]

            database_img_aligned = self.page_state.db_frames[db_idx]

            xyxy_query = list(filter(self.is_query, st.session_state.result))
            xyxy_db = list(filter(self.is_db, st.session_state.result))

            
            xyxy_query = [item['bboxes'] for item in xyxy_query]
            xyxy_db = [item['bboxes'] for item in xyxy_db]
            
            self._segment_annotation(
                input_img=query_img_orig,
                output_file=QUERY_ANNOTATE_IMG,
                xyxy=xyxy_query,
                isQuery=True,
            )

            self._segment_annotation(
                input_img=database_img_aligned,
                output_file=DB_ANNOTATE_IMG, 
                xyxy=xyxy_db,
                isQuery=False,
            )

            self.page_state.annotated_frame = {
                "query": QUERY_ANNOTATE_IMG,
                "db": DB_ANNOTATE_IMG,
                "bboxes_query": xyxy_query,
                "labels_query": self.page_state.annotated_frame['labels_query'],
                "bboxes_db": xyxy_db,
                "labels_db": self.page_state.annotated_frame['labels_db'],
                "label_list": self.page_state.annotated_frame['label_list'],
                "mask_query": self.mask_query,
                "mask_db" : self.mask_db,
                "idx": self.page_state[PlaybackController.name][SLIDER_KEY],
            }
            st.session_state.new_segment = True




    def save_annotation(self):
        detector: ObjectDetectionEngine = self.page_state[self.name][DETECTOR_KEY]
        match: Match = self.page_state.matches[self.page_state[PLAY_IDX_KEY]]
        query_idx: QueryIdx = match[0]
        db_idx: DatabaseIdx = match[1]
        if self.page_state.warp:
            query_img_orig = self.page_state.warp_query_frames[query_idx]
        else:
            query_img_orig = self.page_state.query_frames[query_idx]

        database_img_aligned = self.page_state.db_frames[db_idx]

        query = list(filter(self.is_query, st.session_state.result))
        db = list(filter(self.is_db, st.session_state.result))
        
        xyxy_query = [{"bbox": item['bboxes'], "label": item['label_names']}  for item in query]
        xyxy_db = [{"bbox": item['bboxes'], "label": item['label_names']} for item in db]

        self._segment_annotation(
                input_img=query_img_orig,
                output_file=QUERY_ANNOTATE_IMG,
                xyxy=xyxy_query,
                isQuery=True,
            )

        self._segment_annotation(
            input_img=database_img_aligned,
            output_file=DB_ANNOTATE_IMG, 
            xyxy=xyxy_db,
            isQuery=False,
        )

        mask_query = list(filter(self.is_query, st.session_state.seg_result))
        mask_db = list(filter(self.is_db, st.session_state.seg_result))
        
        mask_query = [item['masks'] for item in mask_query]
        mask_db = [item['masks'] for item in mask_db]
        
        mask_image = detector.save_segmented_masks(self.mask_query, self.mask_db, "")

        # Function to encode an image to base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:  
                return base64.b64encode(image_file.read()).decode('utf-8')  
            
        base64_query = encode_image(query_img_orig)
        base64_db = encode_image(database_img_aligned)

        ''' 
        json save format:
        "image_query": original query image (base64)
        "image_db": original database image (base64)
        "bboxes_query": bounding box annotations for query image
        "bboxes_db": bounding box annotations for database image
        "mask_query": 2D list with query masks
        "mask_db": 2D list with database masks
        "annotated_query": query image with masks and bounding boxes printed onto the image 
        "annotated_db": database image with masks and bounding boxes printed onto the image
        "mask_combined": 2D array with both query and database masks combined
        '''

        # NOTE: some fields are commented out for LLM query testing purposes
        # See GPT.ipynb for information
        save_json = {
                        "image_query": base64_query,
                        "image_db": base64_db,
                        "bboxes_query": xyxy_query, 
                        "bboxes_db": xyxy_db,
                        # "mask_query": self.mask_query.tolist(),
                        # "mask_db": self.mask_db.tolist(),
                        # "annotated_query": self.annotated_image_query.tolist(),
                        # "annotated_db": self.annotated_image_db.tolist(),
                        # "mask_combined": mask_image.tolist(),
                    }
        
        # save format: {video_title}_{query_frame_idx}_{db_frame_idx}
        with open(f"{ANNOTATE_SAVE_PATH}/quetzal_annotated_{query_idx}_{db_idx}.json", "w") as outfile:
            json.dump(save_json, outfile)

    
    def render_prompt(self):
        c1, c2 = st.columns([100, 1])

        with c1:
            with stylable_container(
                key="class_prompt_list",
                css_styles="""{
                    display: block;
                    padding: 0.5em 0.5em 0em; /*top right&left bottom*/
                    width: calc(101% - 133px);
                    }
                """,
            ):
                with st.container():
                    st_tags(
                        label=" ",
                        text="Detection Prompt: Press enter to add more class",
                        value=ss[CLASS_PROMPT_KEY],
                        suggestions=[],
                        maxtags=10,
                        key=CLASS_PROMPT_KEY,
                    )

        with c2:
            with stylable_container(
                key="object_annotate_button",
                css_styles="""{
                        display: block;
                        position: absolute;
                        width: 133px !important;
                        right: 0px;
                        
                        & div {
                            width: 133px !important;
                            height: auto; 
                        }
                        
                        & iframe {
                            width: 133px !important;
                            height: 57px;
                        }                    
                    }
                    """,
            ):
                with elements("object_annotation_controller"):
                    mui.Button(
                        children="Detect",
                        variant="contained",
                        startIcon=mui.icon.Search(),
                        onClick=self.run_detection,
                        size="large",
                        sx={
                            "bgcolor": "grey.800",
                            "borderRadius": "0.5rem",
                            "width": "117.14px",
                        },
                    )

    def change_model(self):
        self.page_state[self.name][DETECTOR_NAME_KEY] = ss[SELECT_MODEL_KEY]
        self.page_state[self.name][DETECTOR_KEY] = getDetectionEngine(
            self.page_state[self.name][DETECTOR_NAME_KEY],
            self.page_state[self.name]["torch_device"],
        )

    def render(self):
        self.render_prompt()

        cc1, cc2, cc3, cc4, cc5 = st.columns([1, 2, 2, 2, 1])
        with cc1:
            st.selectbox(
                label="Choose Detection Model",
                options=[key for key in detector_dict.keys()],
                key=SELECT_MODEL_KEY,
                index=self.select_dict[self.page_state[self.name][DETECTOR_NAME_KEY]],
                on_change=self.change_model,
            )
        with cc2:
            st.slider(
                value=self.page_state[ObjectAnnotationController.name][SLIDER_BOX_TH_KEY],
                label="Box Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format=f"%0.2f",
                key=SLIDER_BOX_TH_KEY,
            )
        with cc3:
            st.slider(
                value=self.page_state[ObjectAnnotationController.name][SLIDER_TXT_TH_KEY],
                label="Text Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format=f"%0.2f",
                key=SLIDER_TXT_TH_KEY,
            )
        with cc4:
            with stylable_container(
                key="object_segment_button",
                css_styles="""{
                        display: block;
                        position: absolute;
                        width: 133px !important;
                        right: 0px;
                        
                        & div {
                            width: 133px !important;
                            height: auto; 
                        }
                        
                        & iframe {
                            width: 133px !important;
                            height: 57px;
                        }                    
                    }
                    """,
            ):
                with elements("segment_button"):
                    mui.Button(
                        children="Segment",
                        variant="contained",
                        startIcon=mui.icon.ContentCut(),
                        onClick=self.segment_annotation,
                        size="large",
                        sx={
                            "bgcolor": "grey.800",
                            "borderRadius": "0.5rem",
                            "width": "117.14px",
                        },
                    )
        with cc5:
            with stylable_container(
                key="object_annotate_save_button",
                css_styles="""{
                        display: block;
                        position: absolute;
                        width: 133px !important;
                        right: 0px;
                        
                        & div {
                            width: 133px !important;
                            height: auto; 
                        }
                        
                        & iframe {
                            width: 133px !important;
                            height: 57px;
                        }                    
                    }
                    """,
            ):
                with elements("save_button"):
                    mui.Button(
                        children="Save",
                        variant="contained",
                        startIcon=mui.icon.Download(),
                        onClick=self.save_annotation,
                        size="large",
                        sx={
                            "bgcolor": "grey.800",
                            "borderRadius": "0.5rem",
                            "width": "117.14px",
                        },
                    )