# C. elegans worm turn detector

# Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Aleksei Romanov, aleksei.a.romanov@gmail.com

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cProfile
from pstats import Stats
import argparse
import traceback
import sys
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
from image_adjustment import apply_brightness_contrast
from datetime import datetime
import longest_chain_cy
from pymediainfo import MediaInfo
import re


def convert_to_black_white_rgb(img_24):
    # Convert to blackwhite, applying contrast adjustment
    res = cv2.cvtColor(img_24, cv2.COLOR_BGR2GRAY)
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    res = apply_brightness_contrast(res, 0, 100)
    res = cv2.bitwise_not(res)

    resv = res
    # cv2.imshow("First", res)

    ret, res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    res_8 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # resT = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # res = cv2.adaptiveThreshold(resT, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 31, -30)
    # res_8 = res

    # ret, res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)
    # res_8 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    return res_8, resv


def get_foreground_mask(
    background_subtractor, morph_kernel, img_24,
):
    foreground_mask_24 = background_subtractor.apply(img_24)

    # cv2.imshow("FG", foreground_mask_24)

    foreground_mask_24 = cv2.morphologyEx(foreground_mask_24, cv2.MORPH_OPEN, morph_kernel,)
    # foreground_mask_24 = cv2.morphologyEx(
    #     foreground_mask_24, cv2.MORPH_DILATE, morph_kernel, iterations=10)
    # foreground_mask_24 = cv2.morphologyEx(
    #     foreground_mask_24, cv2.MORPH_ERODE, morph_kernel, iterations=10)

    (ret, foreground_mask_8,) = cv2.threshold(foreground_mask_24, 0, 255, cv2.THRESH_BINARY,)
    return foreground_mask_8


class VideoReader:
    def __init__(self, video_name):
        self._video_name = video_name
        self._cap = None
        self.total_frames = 0
        self.frame_idx = 0

    def __del__(self):
        if self._cap != None:
            self._cap.release()

    def read_frame(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(self._video_name)
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_idx = 0

        if self._cap == None:
            return False, None

        ret, frame_24 = self._cap.read()
        self.frame_idx += 1

        return ret, frame_24


def point_to_line_distance(pnt, start, line):
    delta = pnt - start
    dist = abs(np.dot(delta, line))

    return dist


class StructureTracker:
    def __init__(self, options):
        self._prev_longest_chain = None
        self._history = []
        self._status_to_accept = 0
        self._turn_count = 0
        self._options = options
        pass

    @property
    def turn_count(self):
        return self._turn_count

    def restore_structure(self, frame_24, gray_8, contour):
        r = cv2.minAreaRect(contour)
        box = cv2.boxPoints(r)
        box = np.int0(box)

        axis1 = np.linalg.norm(box[0] - box[1])
        axis2 = np.linalg.norm(box[0] - box[3])

        v1 = np.array([0, 0])
        v2 = np.array([0, 0])

        minx = min(box[:, 0])
        miny = min(box[:, 1])
        maxx = max(box[:, 0])
        maxy = max(box[:, 1])

        img_roi = gray_8[miny:maxy, minx:maxx]
        img_roi = np.zeros(img_roi.shape)
        cv2.drawContours(
            img_roi, [contour - (minx, miny)], 0, 255, -1,
        )
        ske = (skeletonize(img_roi // 255) * 255).astype(np.uint8)

        # if self._options.debug_vis:
        #     cv2.imshow("interest area", np.hstack((img_roi, ske)))

        if axis1 < axis2:
            v1 = box[0]
            v2 = box[1]
        else:
            v1 = box[0]
            v2 = box[3]

        list_of_non_zero_idx = np.transpose(np.nonzero(ske))
        if len(list_of_non_zero_idx) == 0:
            return
        list_of_non_zero_idx = np.flip(list_of_non_zero_idx, 1)
        list_of_non_zero_idx = list_of_non_zero_idx + np.array([minx, miny])

        line = v2 - v1
        line = line / np.linalg.norm(line)
        line = np.array([line[1], -line[0]])
        line = np.transpose(line)

        distances = np.array([point_to_line_distance(idx, v1, line) for idx in list_of_non_zero_idx])

        res = np.argmin(distances)
        minidx = list_of_non_zero_idx[res]

        longest_chain = longest_chain_cy.longest_chain(ske, int(minidx[1]), int(minidx[0]), int(miny), int(minx),)

        # Determine ordering
        if self._prev_longest_chain is not None:
            prev_len = len(self._prev_longest_chain)
            cur_len = len(longest_chain)

            dist_ordinary_order = 0
            dist_reversed_order = 0

            for i in range(0, min(prev_len, cur_len),):
                p0 = np.array([longest_chain[i][0], longest_chain[i][1],])
                p1 = np.array([self._prev_longest_chain[i][0], self._prev_longest_chain[i][1],])
                p1_rev = np.array([self._prev_longest_chain[-1 - i][0], self._prev_longest_chain[-1 - i][1],])

                dist_ordinary_order += np.dot(p0 - p1, p0 - p1)
                dist_reversed_order += np.dot(p0 - p1_rev, p0 - p1_rev,)

            if dist_reversed_order < dist_ordinary_order:
                longest_chain.reverse()

        self._prev_longest_chain = longest_chain

        p_begin = longest_chain[0]
        p_end = longest_chain[-1]

        if len(longest_chain) >= 3:
            p_mid = longest_chain[int(len(longest_chain) / 2)]
            cv2.line(
                frame_24, (p_begin[1], p_begin[0],), (p_mid[1], p_mid[0]), (255, 0, 0), 2,
            )
            cv2.line(
                frame_24, (p_mid[1], p_mid[0]), (p_end[1], p_end[0]), (255, 0, 0), 2,
            )

            cv2.circle(
                frame_24, (p_begin[1], p_begin[0],), 5, (0, 0, 255), -1,
            )
            cv2.circle(
                frame_24, (p_mid[1], p_mid[0]), 5, (0, 255, 0), -1,
            )
            cv2.circle(
                frame_24, (p_end[1], p_end[0]), 5, (255, 0, 0), -1,
            )

            def angle_between_2(a, b):
                v1 = a / np.linalg.norm(a)
                v2 = b / np.linalg.norm(b)

                cosTh1 = np.dot(v1, v2)
                sinTh1 = v1[0] * v2[1] - v1[1] * v2[0]

                sign = 1
                if sinTh1 < 0:
                    sign = -1

                cosTh1 = np.clip(cosTh1, -1, 1)

                angle_rad = math.acos(cosTh1) * sign
                angle_deg = math.degrees(angle_rad)

                return angle_deg

            np_p_begin = np.array([p_begin[0], p_begin[1]])
            np_p_mid = np.array([p_mid[0], p_mid[1]])
            np_p_end = np.array([p_end[0], p_end[1]])

            v1 = np_p_mid - np_p_begin
            v2 = np_p_end - np_p_mid

            angle_deg = angle_between_2(v1, v2)

            eps = 5

            status = 0
            if angle_deg < 0 - eps:
                status = -1
            elif angle_deg > 0 + eps:
                status = 1
            else:
                status = 0

            if len(self._history) == 0:
                self._history.append(status)
                if status != 0:
                    self._status_to_accept = status
            else:
                last_status = self._history[-1]
                if last_status != status:
                    self._history.append(status)

                    if status != 0:
                        if self._status_to_accept == 0:
                            self._status_to_accept = status
                            self._turn_count += 1
                        else:
                            if status != self._status_to_accept:
                                self._status_to_accept = 0
                            else:
                                self._turn_count += 1


class SingleWormDetector:
    def __init__(self, options, video_name):
        self._video_name = video_name
        self._options = options
        self._frames_processed = 0
        pass

    def detect_turns(self):
        video = VideoReader(self._video_name)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        background_subtractor = cv2.createBackgroundSubtractorMOG2()
        # background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)

        paused = False

        # escape
        escape_key = 27

        # pause - whitespace
        pause_key = 32

        structure_tracker = StructureTracker(options)

        show_gui = self._options.debug_vis or self._options.show_frames

        while True:
            if show_gui:
                if paused == True:
                    k = cv2.waitKey(30) & 0xFF
                    if k == escape_key:
                        return structure_tracker.turn_count
                    if k == pause_key:
                        paused = False
                    continue

            (ret, frame_24,) = video.read_frame()

            if ret == False:
                return structure_tracker.turn_count

            target_width = 550

            if frame_24.shape[1] > target_width:
                scale = target_width / frame_24.shape[1]
                width = int(frame_24.shape[1] * scale)
                height = int(frame_24.shape[0] * scale)
                dim = (width, height)
                frame_24 = cv2.resize(frame_24, dim, interpolation=cv2.INTER_NEAREST,)

            # Grayscale + color curvers + threshold
            (gray_8, resv,) = convert_to_black_white_rgb(frame_24)

            # Foreground mask
            fg_mask_8 = get_foreground_mask(background_subtractor, kernel, frame_24,)

            edges = cv2.Canny(image=frame_24, threshold1=100, threshold2=200,)
            edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=2,)
            # gray_8 = edges
            cv2.cvtColor(
                edges, cv2.COLOR_GRAY2BGR,
            )
            cv2.imshow("edges", edges)

            # Get all contours
            (contours, hierarchy,) = cv2.findContours(gray_8, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS,)

            rendered_contours_frame_8 = np.zeros((frame_24.shape[0], frame_24.shape[1],), dtype=np.uint8,)
            for cnt in contours:
                cv2.drawContours(
                    rendered_contours_frame_8, [cnt], -1, (255), thickness=cv2.FILLED,
                )

            # Overlap contours and forgeground mask
            contours_and_fg_mask_8 = cv2.bitwise_and(fg_mask_8, rendered_contours_frame_8, 255,)

            indices_of_covered_points = np.transpose(np.nonzero(contours_and_fg_mask_8))

            covered_pts = []

            # Indices of covered points
            for idx in indices_of_covered_points:
                covered_pts.append((int(idx[1]), int(idx[0]),))

            def get_contour_with_max_coverage_idx(
                gray, contours, covered_pts,
            ):
                img_roi = np.zeros(gray.shape)
                idx = 0

                max_countour_idx = -1
                for cnt in contours:
                    ref_color = idx
                    cv2.drawContours(
                        img_roi, [cnt], 0, ref_color, -1,
                    )
                    idx += 1

                res = {}

                for pt in covered_pts:
                    pT = (
                        int(pt[1]),
                        int(pt[0]),
                    )
                    clr = img_roi[pT]
                    if clr in res:
                        res[clr] += 1
                    else:
                        res[clr] = 1

                if len(res) > 0:
                    max_countour_idx = int(max(res, key=res.get,))

                return max_countour_idx

            if len(covered_pts) > 0 and len(contours) > 0:
                contour_with_max_points_idx = get_contour_with_max_coverage_idx(gray_8, contours, covered_pts,)
                if contour_with_max_points_idx != -1:
                    largest_covering_contour = contours[contour_with_max_points_idx]
                    # cv2.drawContours(largets_covering_frame_8, [largest_covering_contour], -1, (255), thickness=cv2.FILLED)
                    structure_tracker.restore_structure(
                        frame_24, gray_8, largest_covering_contour,
                    )

            prev_percent = int(100 * self._frames_processed / video.total_frames)
            cur_percent = int(100 * video.frame_idx / video.total_frames)
            if prev_percent != cur_percent:
                print(f"frames processed: {cur_percent}% - {video.frame_idx}/{video.total_frames}")
            self._frames_processed = video.frame_idx

            if show_gui:
                cv2.putText(
                    frame_24, f"Turn Count:{structure_tracker.turn_count}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                )

                gray_24 = cv2.cvtColor(gray_8, cv2.COLOR_GRAY2BGR,)
                cv2.putText(
                    gray_24, "B/W", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                )

                fg_mask_24 = cv2.cvtColor(fg_mask_8, cv2.COLOR_GRAY2BGR,)
                cv2.putText(
                    fg_mask_24, "FG mask", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                )

                images_to_show = frame_24
                if self._options.debug_vis:
                    images_to_show = np.hstack((frame_24, gray_24, fg_mask_24,))

                cv2.imshow(
                    "worm_turn", images_to_show,
                )

                cv2.moveWindow("worm_turn", 0, 0)
                k = cv2.waitKey(1) & 0xFF
                if k == escape_key:
                    return structure_tracker.turn_count
                if k == pause_key:
                    paused = True


def get_videolist():
    video_files = []

    files = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f)]
    for f in files:
        media_info = MediaInfo.parse(f)
        for track in media_info.tracks:
            if track.track_type == "Video":
                video_files.append(f)
                break

    return video_files


def compare_against_ground_truth(results_filename,):
    ground_truth_compared = {}

    try:
        with open("ground_truth.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                m = re.search("(.*) (\d+)", line)
                if m is None:
                    continue

                name = m.group(1).lower()
                num = int(m.group(2))
                ground_truth_compared[name] = [num, 0]

        with open(results_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                m = re.search("Video: '(.*)', Turns: (\d+)", line,)
                if m is None:
                    continue

                name = m.group(1).lower()
                (fname, fext,) = os.path.splitext(os.path.basename(name))
                name = fname.lower()
                num = int(m.group(2))

                if name in ground_truth_compared:
                    ground_truth_compared[name][1] = num

        with open(f"compared_{results_filename}", "w",) as f:
            for (name, comparison,) in ground_truth_compared.items():
                if comparison[1] == 0:
                    continue
                error = abs(int(100 * float(comparison[1] - comparison[0]) / comparison[0]))
                f.write(f"{name}, {comparison[0]}/{comparison[1]}, {error}%\n")

    except:
        return


def main(options, date_prefix):
    video_list = get_videolist()
    print("Videos to process")
    for idx, video in enumerate(video_list):
        print(f"#{idx} - '{video}'")

    detected_turns_list = []

    for video in video_list:
        print(f"Processing video: '{video}'")

        detector = SingleWormDetector(options, video)

        turns_count = detector.detect_turns()
        if options.debug_vis:
            cv2.destroyAllWindows()

        detected_turns_list.append(turns_count)

        print(f"Detected {turns_count} turns")

    results_filename = f"{date_prefix}_results.txt"

    with open(results_filename, "a+") as f:
        for v, t in zip(video_list, detected_turns_list,):
            f.write(f"Video: '{v}', Turns: {t}\n")

    compare_against_ground_truth(results_filename)


def get_options():
    parser = argparse.ArgumentParser(description="Worm turn detector, Aleksei Romanov, aleksei.a.romanov@gmail.com")
    parser.add_argument(
        "--show_frames", action="store_true", default=False, help="Visualize worm structure and turns count realtime",
    )
    parser.add_argument(
        "--debug_vis", action="store_true", default=False, help="Enable debug visualization",
    )
    parser.add_argument(
        "--dump_stats", action="store_true", default=False, help="Enable statistics dump",
    )

    options = parser.parse_args()
    return options


if __name__ == "__main__":
    print("C. elegans worm turn detector. Aleksei Romanov, aleksei.a.romanov@gmail.com")

    print(f"Using python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    date_prefix = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")

    try:
        options = get_options()

        profiler = cProfile.Profile()
        if options.dump_stats:
            profiler.enable()

        main(options, date_prefix)

        if options.dump_stats:
            profiler.disable()
            stats = Stats(profiler)
            stats.strip_dirs()
            stats.dump_stats(f"{date_prefix}.prof_stats")

    except Exception as err:
        print(err)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
