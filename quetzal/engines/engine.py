#!/usr/bin/env python

from abc import ABC, abstractmethod

class AbstractEngine(ABC):
    # @abstractmethod
    # def warmup(self):
    #     '''Initial warm-up if needed
    #     '''
    # pass
    name = "Default Name"

    @staticmethod
    def is_video_analyzed(video) -> bool:
        """Return True if no further real-time analysis required"""

        return False

    @abstractmethod
    def analyze_video(self, video):
        """Return True if no further real-time analysis required"""

        pass

    @abstractmethod
    def process(self, file_path: list):
        """Process list of files in file_path

        Return an resulting file."""

        pass

    @abstractmethod
    def end(self):
        """Indicate no more input will be processed"""
        pass

    @abstractmethod
    def save_state(self, save_path):
        """Save state in save_path. return None or final result"""
        pass


class ObjectDetectionEngine(ABC):
    name = "Default Name"

    def __init__(self, device):
        pass

    @staticmethod
    def is_video_analyzed(video) -> bool:
        """Return True if no further real-time analysis required"""

        return False

    @abstractmethod
    def generate_masked_images(
        self, query_image, caption, save_file_path, box_threshold, text_threshold
    ):
        pass
