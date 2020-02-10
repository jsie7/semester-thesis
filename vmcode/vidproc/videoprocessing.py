#!/usr/bin/env python

"""Video processor module.

This module processes any video type and loads the corresponding labels given
in a eaf file.
The module cuts the video into frames, processes the frames according to the
process and post-process operations defined by the user and loads the labels
for each frame.

Process Operations:
    The process operations are defined as class-level methods inside the
    VideoProcessor class and are private by default. The user can add any
    process operation seen fit, but the method should follow the layout below
    to guarantee compatability.

    def _process_method(self, mode, X, options=any_option_string):
        First the method is called in initialization mode and subsequently
        in run mode on each frame.

        Note:
            The video is passed to this method frame by frame and not a
            whole video!

        if mode == 'init':
            - process option string and assert correct format.
            - write any meaningful output to console.
            - compute shape of output array.

            Returns:
                [[(shape of output array), output array dtype],
                 processed option string]

        elif mode == 'run':
            - process the frame.

            Returns:
                Processed frame with shape and dtype defined by 'init' mode.

        else:
            raise RuntimeError("Invalid mode {0} for this operation!"
                               .format(mode))

Post-Process Operations:
    Post-process operations are defined as module-level methods right below
    this docstring. The operations are declared public, so they can be
    imported from the module. This allows to use them on already processed
    data.

    Note:
        The post-process methods can also be used to further process data
        after it has been cached. E.g. process data to a 'close to final'
        form, cache the data, then process the data to final form and
        return the data. This is useful if the 'close to final' data form
        is far more efficient to store than the final data form itself.

    The user can add any post-process operation seen fit, but the method
    should follow the layout below to guarantee compatability.

    def postprocess_operation(X, options):
        This method doesn't get the data frame by frame but the whole array at
        once.

        Args:
            X (np.ndarray): Processed array of all video frames.
            options (list): The processed options returned by the 'init' mode
                of the process method.

        Returns:
            The post-processed array.

Todo:
    * What happens if ts_ref1[k+1] != ts_ref2[k]?
    * Add support for string labels.
    * Implement subsampling.
    * Add functionality to only load certain data columns from cache.
    * Replace process and post-process operations and move them to a separate
      file, e.g. see turbidity_args.py. This can be used via a YAML config
      file as well.

"""

# built-in modules
import datetime as dt
import logging
import os
import warnings

# third party modules
import cv2
import numpy as np
import scipy.sparse as sparse
# import h5py and suppress FutureWarning (flake8 will complain!)
np.warnings.filterwarnings("ignore", category=FutureWarning)
import h5py

# project specific modules
from vmcode.recproc import eaf_processing as eaf

# set logging configuration
logger = logging.getLogger("VideoProcessor")
logging.basicConfig(format='%(name)s - %(levelname)s: %(message)s',
    level=logging.INFO)


""" START OF POST-PROCESSING OPERATIONS SECTION """


def postprocess_hist(X, options):
    """Post-process method for 1D histograms.

    Args:
        X (np.ndarray): 1D-histograms of frames as a 2D-array.
        options (list): Parameters used to calculate bins: Letter indicating
            which layer of the HSV color space was used; Integer indicating
            the number of bins used.

    Returns:
        np.ndarray: Normalized 1D-histograms (empirical probability density
            function).

    """
    hsv_dict = {'h': 180, 's': 256, 'v': 256}
    assert len(options) == 2
    assert isinstance(options[0], str)
    assert isinstance(options[1], int)

    denum = X.sum(axis=1)
    denum[np.where(denum == 0)] = 1.0
    return X/(denum*(hsv_dict[options[0]]-1)/options[1]).reshape(-1, 1)


def postprocess_hist2d(X, options):
    """Post-process method for 2D histograms.

    Args:
        X (np.ndarray): 2D-histograms of frames as a 3D-array.
        options (list): Parameters used to calculate bins: 2 letters
            indicating which layers of the HSV color space were used;
            2 integers indicating the number of bins used.

    Returns:
        np.ndarray: Normalized 2D-histograms (empirical 2D probability density
            function).

    """
    hsv_dict = {'h': 180, 's': 256, 'v': 256}
    assert len(options) == 4
    assert isinstance(options[0], str) and isinstance(options[2], str)
    assert isinstance(options[1], int) and isinstance(options[3], int)
    assert len(X.shape) == 3

    denum = X.sum(axis=(1, 2))
    denum[np.where(denum == 0)] = 1.0
    return X/(denum*((hsv_dict[options[0]]-1)/options[1])*(
            (hsv_dict[options[2]]-1)/options[3])).reshape(-1, 1, 1)


""" END OF POST-PROCESSING OPERATIONS SECTION """


class VideoProcessor(object):
    """Class to process video and annotation files.

        This class loads the video and the annotations from file and
        subsequently processes the video frame by frame.

        Example:
            v = VideoProcessor("path_to_video_file", caching="exact",
                                cache_location="cache")
            X,y = v.process(process_options=
                  "process_options_string_according_to_user_defined_methods",
                  label_tier="Tier_name_in_annotation_file")
            or:
            X,y = load_from_cache("path_to_video_file", "cache")

        Note:
            The user-defined process methods are defined in the process method
            section right after the __init__ method.

    """
    def __init__(self, vid_fname, label_fname=None, caching=None,
                 cache_location=None, cache_instance=None, timestamp=None):
        """Initialize VideoProcessor class.

        Args:
            vid_fname (str): Path to video file as string.
            label_fname (str): Path to annotation file as string. If
                label_fname is 'None', the folder containing the video file is
                searched to find an eaf file to use for label generation.
            caching (str): If 'None' processed data won't be cached.
                If 'exact' the processed data will be cached to cache_location
                in an uncompressed format. If 'compressed' the processed data
                will be cached to cache_location in a compressed format
                specified in the process dictionary.
            cache_location (str): Path to cache location. This is omitted if
                caching==None.
            cache_instance (str): The cache folder structure is as follows:
                "cache/video_file_name/cache_instance". If 'None' the instance
                name will be set to the current timestamp. The instance
                name can be used to load a specific data set from cache.
            timestamp (int or str): Timestamp. If 'None' a timestamp will be
                created during initialization.

        Raises:
            NameError: If opencv couldn't be imported.
            RuntimeError: If the video file couldn't be opened by opencv.
            RuntimeError: If the path to the cache location is not valid.
            RuntimeError: If an invalid option is passed to caching.
            Warning (No label file found!): If lable_fname=None and no eaf
                file is found in the folder containing the video file.

        """
        logger.info("----- VIDEO PROCESSOR -----")
        try:
            logger.info("Using opencv version {0}".format(cv2.__version__))
        except NameError as err:
            raise NameError("opencv not available! Install it, before "
                            "using this module.")

        self._vid_fname = vid_fname
        self._label_fname = label_fname

        if self._label_fname is None:
            dir = os.path.dirname(self._vid_fname)
            for file in os.listdir(dir):
                if file.endswith(".eaf"):
                    f = os.path.join(dir, file)
                    if os.path.isfile(f):
                        self._label_fname = f
                    else:
                        warnings.warn("No label file found!", RuntimeWarning)
                        self._label_fname = None

        self.vid = cv2.VideoCapture(self._vid_fname)
        if not self.vid.isOpened():
            raise RuntimeError("Video file at {0} is not open!".format(
                                self._vid_fname))
        self._nr_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self._height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        logger.info("Load video: {0}".format(self._vid_fname))
        logger.info(("Video duration: {0} s, frame rate: {1} fps, number of "
              "frames: {2}").format(self._nr_frames/self._fps, self._fps,
                                    self._nr_frames))

        if caching is not None:
            self._cache = True
            if cache_location is not None and os.path.isdir(cache_location):
                self._cache_location = cache_location
            else:
                raise RuntimeError("'{0}' is not a valid cache location!"
                                   .format(cache_location))

            if caching == "exact":
                self._compress_cache = False
            elif caching == "compressed":
                self._compress_cache = True
            else:
                raise RuntimeError("Invalid option for caching!")
        else:
            self._cache = False
            self._cache_location = None
            self._compress_cache = False

        self._cache_instance = cache_instance
        if timestamp is None:
            timestamp = int(dt.datetime.now().timestamp())
        if not isinstance(timestamp, int):
            self.timestamp = int(timestamp)
        else:
            self.timestamp = timestamp

    """ START OF OPERATIONS SECTION """

    def _raw_operation(self, mode, X, options=None):
        """Raw operation method.

        Does no processing, only returns the raw frames.

        Example:
            Use "raw" in process_options string.

        Args:
            None.

        Returns:
            np.ndarray: Raw frames of size (frame_height, frame_width, 3).

        Note:
            This option will return a huge array, possibly filling all of the
            available RAM!

        """
        if mode == 'init':
            logger.info("  - Returning raw frames...")
            if self._cache:
                warnings.warn("Shouldn't cache raw frames! This will use "
                              "too much of disk space! Setting caching to "
                              "'False'...")
                self._cache = False
            return [[(self._nr_frames, self._height, self._width, 3),
                    np.uint8], None]
        elif mode == 'run':
            return X
        else:
            raise RuntimeError("Invalid mode {0} for this operation!".format(
                mode))

    def _redness_operation(self, mode, X, options=180):
        """Redness operation method.

        Calculates the ratio of red pixels in all pixels, by using
        a 1D histogram of the Hue values of the frame.

        Example:
            Use "redness(:arg1)" in process_options string.

        Args:
            options (int): Number of bins used to calculate histogram
                of Hue values.

        Returns:
            np.ndarray: Processed frames of size (1).

        Note:
            This method maskes the frames first to get rid of the black frame
            border.

        """
        if mode == 'init':
            if not isinstance(options, int):
                assert len(options) == 1
                redness_options = int(options[0])
            else:
                redness_options = options
            logger.info("  - Computing redness of frames with {0} initial"
                " bins...".format(redness_options))
            return [[(self._nr_frames, 1), np.float64], redness_options]
        elif mode == 'run':
            # mask frame to get rid of black surrounding
            hsv = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 0]),
                               np.array([255, 255, 1]))

            row = np.bincount(hsv[~mask.astype(bool), 0], minlength=options)
            denum = row.sum()
            if denum == 0:
                return 0
            else:
                return (1 - row[int(options/18):-int(options/18)].sum()/denum)
        else:
            raise RuntimeError("Invalid mode {0} for this operation!"
                               .format(mode))

    def _hist_operation(self, mode, X, options=['h', 180]):
        """1D histogram operation method.

        Calculates the 1D histogram over the specified layer of the HSV
        color space.

        Example:
            Use "hist:arg1(:arg2)" in process_options string.

        Args:
            options (list): The letter specifies the layer of the HSV color
                space used; The integer specifies the number of bins used.

        Returns:
            np.ndarray: Processed frames of size (nr_bins).

        Note:
            - This method maskes the frames first, to get rid of the black
              frame border.
            - This method needs a post-process method to normalize the
              histograms.
            - If cached exactly, the unnormalized histograms are stored and
              normalized during retrieval from cache; If cached with
              compression, the normalized histograms are stored as numpy's
              float16 data type.

        """
        hsv_dict = {'h': [180, 0], 's': [256, 1], 'v': [256, 2]}
        if mode == 'init':
            assert len(options) <= 2, "Too many options for this " + \
                "operation! Maximum is 2."
            if len(options) == 1:
                assert (options[0] == 'h' or options[0] == 's' or
                        options[0] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                hist_options = [options[0], hsv_dict[options[0]][0]]
            else:
                assert (options[0] == 'h' or options[0] == 's' or
                        options[0] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                hist_options = [options[0], int(options[1])]
            logger.info("  - Computing 1D histogram of '{a[0]}' with {a[1]}"
                " bins...".format(a=hist_options))
            return [[(self._nr_frames, hist_options[1]), np.uint32],
                    hist_options]
        elif mode == 'run':
            # mask frame to get rid of black surrounding
            hsv = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 0]),
                               np.array([255, 255, 1]))

            if options[1] == hsv_dict[options[0]][0]:
                row = np.bincount(hsv[~mask.astype(bool),
                                  hsv_dict[options[0]][1]],
                                  minlength=options[1])
                denum = row.sum()
                if denum == 0:
                    return np.zeros(options[1])
                else:
                    return row
            else:
                ind = np.floor(hsv[~mask.astype(bool),
                               hsv_dict[options[0]][1]].astype(float) *
                               options[1]/(hsv_dict[options[0]][0]-1))
                row = np.bincount(ind.astype(int), minlength=options[1])
                if row.shape[0] > options[1]:
                    row[-2] += row[-1]
                    row = row[:-1]
                denum = row.sum()
                if denum == 0:
                    return np.zeros(options[1])
                else:
                    return row
        else:
            raise RuntimeError("Invalid mode {0} for this operation!"
                               .format(mode))

    def _hist2d_operation(self, mode, X, options=['h', 180, 's', 256]):
        """2D histogram operation method.

        Calculates the 2D histogram over the specified layers of the HSV
        color space.

        Example:
            Use "hist2d:arg1:arg2(:arg3:arg4)" in process_options string.

        Args:
            options (list): The 2 letters specify the layers of the HSV color
                space used; The 2 integers specify the number of bins used.

        Returns:
            np.ndarray: Processed frames of size (nr_bins1, nr_bins2).

        Note:
            - This method maskes the frames first, to get rid of the black
              frame border.
            - This method needs a post-process method to normalize the
              histograms.
            - If cached exactly, the unnormalized histograms are stored and
              normalized during retrieval from cache; If cached with
              compression, the normalized histograms are stored as numpy's
              float16 data type.

        """
        hsv_dict = {'h': [180, 0], 's': [256, 1], 'v': [256, 2]}
        if mode == 'init':
            assert len(options) <= 4, ("Too many options for this "
                                       "operation! Maximum is 4.")
            if len(options) == 2:
                assert (options[0] == 'h' or options[0] == 's' or
                        options[0] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                assert (options[1] == 'h' or options[1] == 's' or
                        options[1] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                hist2d_options = [options[0], hsv_dict[options[0]][0],
                                  options[1], hsv_dict[options[1]][0]]
            else:
                assert (options[0] == 'h' or options[0] == 's' or
                        options[0] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                assert (options[2] == 'h' or options[2] == 's' or
                        options[2] == 'v'), ("Not a valid option for this "
                                             "process! Use 'h', 's' or 'v'.")
                hist2d_options = [options[0], int(options[1]), options[2],
                                  int(options[3])]
            logger.info(("  - Computing 2D histogram of '{a[0]}' and '{a[2]}' "
                  "with {a[1]} and {a[3]} bins...").format(a=hist2d_options))
            return [[(self._nr_frames, hist2d_options[1], hist2d_options[3]),
                    np.uint32], hist2d_options]
        elif mode == 'run':
            # mask frame to get rid of black surrounding
            hsv = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 0, 0]),
                               np.array([255, 255, 1]))
            if (options[1] == hsv_dict[options[0]][0] and options[3] ==
                    hsv_dict[options[2]][0]):
                row = hsv[~mask.astype(bool), hsv_dict[options[0]][1]]
                col = hsv[~mask.astype(bool), hsv_dict[options[2]][1]]

                if row.shape[0] == 0 or col.shape[0] == 0:
                    return np.zeros((options[1], options[3]))
                else:
                    weights = np.ones(row.shape[0])
                    return sparse.coo_matrix((weights, (row, col)), shape=(
                        options[1], options[3])).toarray()
            else:
                row = np.floor(hsv[~mask.astype(bool), hsv_dict[
                    options[0]][1]].astype(float)*options[1]/(
                    hsv_dict[options[0]][0]-1))
                col = np.floor(hsv[~mask.astype(bool), hsv_dict[
                    options[2]][1]].astype(float)*options[3]/(
                    hsv_dict[options[2]][0]-1))

                if row.shape[0] == 0 or col.shape[0] == 0:
                    return np.zeros((options[1], options[3]))
                else:
                    if np.amax(row) > (options[1]-1):
                        row[np.where(row == options[1])] = options[1]-1
                    if np.amax(col) > (options[3]-1):
                        col[np.where(col == options[3])] = options[3]-1
                    weights = np.ones(row.shape[0])
                    return sparse.coo_matrix((weights, (row, col)), shape=(
                        options[1], options[3])).toarray()
        else:
            raise RuntimeError("Invalid mode {0} for this operation!"
                               .format(mode))

    def _default_operation(self, mode, X, options=None):
        """Default operation method.

        Does nothing. Fallback method if no or an invalid process operation
        is selected.

        Example:
            Use "default" or 'None' in process_options string.

        Args:
            None.

        Returns:
            np.ndarray: Empty numpy array.

        """
        if mode == 'init':
            logger.info("  - No process was selected. Nothing done!")
            self._cache = False
            return [[(1, 1), np.object], None]
        elif mode == 'run':
            if self.vid.isOpened():
                self.vid.release()
            return None
        else:
            raise RuntimeError("Invalid mode {0} for this operation!"
                               .format(mode))

    """ END OF OPERATIONS SECTION """

    def _process_labels(self):
        """Method to process annotation files.

        Opens edf file and loads labels of tier specified in the __init__
        method. Writes labels to class attribute y.

        Note:
            This method will be called by the process method.

        """
        if self._label_tier is None:
            self._raw_labels = None
            self.y = None
            logger.info("Using unlabeled data")
        else:
            self._raw_labels = eaf.get_eaf_tier_as_df(self._label_fname,
                                                      self._label_tier)
            assert self._label_tier in self._raw_labels.columns, (
                   "There is no tier with label: {0}"
                   .format(self._label_tier))
            assert "ts_ref1" in self._raw_labels
            self.y = np.empty(self._nr_frames)
            self._index = (self._raw_labels.ts_ref1.astype(int) /
                           (1000/self._fps)).round().astype(int)
            if self._index[0] != 0:
                self._index[0] = 0
            for i, ind in enumerate(self._index):
                if i+1 == len(self._index):
                    self.y[ind:] = self._raw_labels[self._label_tier][i]
                else:
                    self.y[ind:self._index[i+1]] = self._raw_labels[
                                                    self._label_tier][i]
            logger.info("Using labeled data")

    def process(self, process_options=None, label_tier=None,
                postprocess=True):
        """Method to process the loaded video and annotation files.

        This method will generate a process list from the options passed
        to it and subsequently iterate through it. In addition, it will cache
        the processed data if caching is specified during initialization.

        Example:
            v = VideoProcessor("path_to_video_file", caching="exact",
                                cache_location="cache")
            X,y = v.process(process_options=
                  "process_options_string_according_to_user_defined_methods",
                  label_tier="Tier_name_in_annotation_file")

        Note:
            This function calls internal operation functions defined in the
            process operation section above this method. Additional operation
            methods can be added to this section, but they have to obey the
            structure outlined at the beginning of this file.

        Args:
            process_options (str): String specifying how to process the video
                file. The standard format is: "option1:arg1_option2:arg1:arg2"
                options are separated by underscores and arguments are
                separated by colons.
                For details on the options, have a look at the process
                methods' docstring.
            label_tier (str): String specifying the tier of the eaf file to
                use as the labels. If None, no labels will be used.
            postprocess (bool): Boolean defining if post-process methods are
                called or not.

        Returns:
            X (list(np.ndarray)): List of numpy arrays. Those arrays contain
                the data processed according to the process_options argument.
                The data is returned in the order, the process options were
                passed to the method.
            y (np.ndarray): Numpy Array containing the processed label data.

        Raises:
            RuntimeError: If the video file couldn't be opened by opencv.
            RuntimeError: If a non specified mode for the operation function
                is chosen.
            RuntimeError: If the folders needed for caching couldn't be
                created.

        """
        self._label_tier = label_tier
        if process_options is None:
            self._process_options = "default"
        else:
            self._process_options = process_options
        self._process_dict = {
            'default': [self._default_operation, None, None],
            'raw': [self._raw_operation, None, None],
            'redness': [self._redness_operation, None, np.float32],
            'hist': [self._hist_operation, postprocess_hist, np.float16],
            'hist2d': [self._hist2d_operation, postprocess_hist2d,
                       np.float16],
            None: [self._default_operation, None, None]
        }

        # PROCESS LABELS
        self._process_labels()

        # PROCESS DATA
        # parse process options
        self._process_list = self._process_options.split("_")
        self._operation_list = []
        self._needs_postprocessing = []
        for process in self._process_list:
            op_detail = process.split(":")
            if len(op_detail) > 1:
                self._operation_list.append([self._process_dict.get(
                    op_detail[0], self._default_operation)[0], op_detail[1:]])
            else:
                self._operation_list.append([self._process_dict.get(
                    op_detail[0], self._default_operation)[0]])
            if self._process_dict.get(op_detail[0],
                                      self._default_operation)[1] is None:
                self._needs_postprocessing.append(False)
            else:
                self._needs_postprocessing.append(postprocess)

        # initialize X
        logger.info("Start processing frames:")
        X_list = []
        self.options_list = []
        for op in self._operation_list:
            if len(op) > 1:
                init = op[0]('init', None, options=op[1])
            else:
                init = op[0]('init', None)
            X_list.append(np.empty(init[0][0], dtype=init[0][1]))
            self.options_list.append(init[1])
        self.X = X_list

        if not self.vid.isOpened():
            raise RuntimeError("Video file at {0} is not open!"
                               .format(self._vid_fname))

        # process video
        i = 0
        while(self.vid.isOpened()):
            ret, frame = self.vid.read()
            if ret:
                for j, op in enumerate(self._operation_list):
                    self.X[j][i] = op[0]('run', frame,
                                         options=self.options_list[j])
                i = i + 1
            else:
                break
        self.vid.release()
        logger.info("Finished processing frames")

        # cache data
        if self._cache:
            logger.info("Start caching...")
            vid_folder_name = self._vid_fname.split("/")[-1].split(".")[0]
            vid_folder_pointer = self._cache_location + "/" + vid_folder_name
            try:
                if not os.path.isdir(vid_folder_pointer):
                    os.mkdir(vid_folder_pointer)
            except:
                raise RuntimeError("Couldn't create folder at '{0}'!"
                                   .format(vid_folder_pointer))

            timestamp = dt.datetime.fromtimestamp(self.timestamp)
            if self._cache_instance is None:
                instance_folder_pointer = vid_folder_pointer + "/" + \
                    timestamp.strftime("%Y%m%d_%H%M%S")
            else:
                assert isinstance(self._cache_instance, str)
                instance_folder_pointer = vid_folder_pointer + "/" + \
                    self._cache_instance

            try:
                if not os.path.isdir(instance_folder_pointer):
                    os.mkdir(instance_folder_pointer)
            except:
                raise RuntimeError("Couldn't create folder at '{0}'!"
                                   .format(instance_folder_pointer))

            info_filename = instance_folder_pointer + '/' + \
                vid_folder_name + '.info'
            h5_filename = instance_folder_pointer + '/' + vid_folder_name + \
                '.h5'

            logger.info("writing h5 file to {0} ...".format(h5_filename))
            with h5py.File(h5_filename, 'w') as hf:
                hf.create_dataset("y",  data=self.y)
                for i, data_object in enumerate(self._process_list):
                    name = data_object.split(":")[0]
                    if self._compress_cache:
                        compress_type = self._process_dict.get(name)[2]
                        postproc = self._process_dict.get(
                                    name, self._default_operation)[1]
                        if (self._needs_postprocessing[i] and postproc is
                                not None):
                            self.X[i] = postproc(self.X[i],
                                                 self.options_list[i])
                            self._needs_postprocessing[i] = False
                        hf.create_dataset(data_object, data=self.X[i].astype(
                            compress_type))
                    else:
                        hf.create_dataset(data_object, data=self.X[i])

            logger.info("writing info file to {0} ...".format(info_filename))
            with open(info_filename, 'w') as f:
                if self._cache_instance is None:
                    f.write("Instance Name: {0} ({1})\n"
                            .format(self._cache_instance,
                                    timestamp.strftime("%Y%m%d_%H%M%S")))
                else:
                    f.write("Instance Name: {0}\n"
                            .format(self._cache_instance))
                f.write("Timestamp: {0}\n"
                        .format(timestamp.strftime("%d.%m.%Y %H:%M:%S")))
                f.write("Data Operations: {0}\n".format(self._process_list))
                f.write("Operation details: {0}\n".format(self.options_list))
                f.write("Needs process if read from cache: {0}\n"
                        .format(self._needs_postprocessing))
                f.write("Compressed: {0}".format(self._compress_cache))
                f.close()

        # postprocess video
        for j, process in enumerate(self._process_list):
            op_detail = process.split(":")
            postproc = self._process_dict.get(op_detail[0],
                                              self._default_operation)[1]
            if self._needs_postprocessing[j] and postproc is not None:
                self.X[j] = postproc(self.X[j], options=self.options_list[j])

        logger.info("----- VIDEO PROCESSOR -----\n")
        return self.X, self.y


def load_from_cache(vid_fname, cache_location, cache_instance=None):
    """Load previously processed data from cache.

    This method extracts data written to the cache by the process method
    of the VideoProcessor class.

    Example:
        X,y = load_from_cache("path_to_video_file", "cache_location")

    Args:
        vid_fname (str): Either path to video file which was previously
            processed or just the name of the video file as a string.
        cache_location (str): Path to the cache location as a string.
        cache_instance (str): Specific instance of cached data; This requires
            that the cache_instance was specified during processing. If
            'None' the method will load the newest cache instance of the
            particular video file, e.g. the data with the newest timestamp.

    Returns:
        X (list(np.ndarray)): List of numpy arrays. Those arrays contain
            the data processed in an earlier call to the process method.
            The data is returned in the order, the process options were
            passed to the process method.
        y (np.ndarray): Numpy array containing the processed label data.

    Raises:
        RuntimeError: If no info and/or h5 file can be found with the
            specified arguments.

    """
    logger.info("----- LOADING FROM CACHE -----")
    process_dict = {
        'raw': None,
        'redness': None,
        'hist': postprocess_hist,
        'hist2d': postprocess_hist2d,
        'default': None
    }

    vid_name = vid_fname.split("/")[-1].split(".")[0]
    folder_pointer = cache_location + "/" + vid_name
    if cache_instance is None:
        folder_list = sorted(os.listdir(folder_pointer), reverse=True)
        for folder in folder_list:
            if folder.startswith("20") and "_" in folder:
                instance = folder
                break
            else:
                instance = 'non-specific instance'
    else:
        instance = cache_instance

    if not os.path.isdir(folder_pointer + "/" + instance):
        raise RuntimeError("Instance '{0}' doesn't exist!"
                           .format(instance))
    file_pointer = folder_pointer + "/" + instance + "/" + vid_name

    if not os.path.isfile(file_pointer + ".info"):
        raise RuntimeError("No valid info file found at '{0}'!"
                           .format(file_pointer + ".info"))
    with open(file_pointer + ".info", 'r') as f:
        instance = f.readline().strip("\n").split(": ")[1]
        timestamp = f.readline().strip("\n").split(": ")[1]
        operation_list = eval(f.readline().strip("\n").split(": ")[1])
        operation_details = eval(f.readline().strip("\n").split(": ")[1])
        needs_process = eval(f.readline().strip("\n").split(": ")[1])
        compressed = eval(f.readline().strip("\n").split(": ")[1])

    if cache_instance is None:
        logger.info(("No specific instance given! Loading most recent "
              "unspecific instance '{0}' of processed video '{1}' from "
              "cache...")
              .format(instance.split(" ")[-1].strip("(").strip(")"),
                      vid_fname.split("/")[-1]))
    else:
        logger.info(("Loading instance '{0}' of processed video '{1}' from "
            "cache ...").format(instance, vid_fname.split("/")[-1]))
    logger.info("The instance was created on {0} with the following "
        "operations:".format(timestamp))
    for i, op in enumerate(operation_list):
        logger.info("  - Operation: '{0}' with options: '{1}'"
              .format(op.split(":")[0], operation_details[i]))
    if np.any(needs_process):
        logger.info("The following operations were processed while loading "
            "from cache:")
        for i, process in enumerate(needs_process):
            if process:
                logger.info("  - '{0}', '{1}'".format(
                    operation_list[i].split(":")[0], operation_details[i]))

    X = []
    if not os.path.isfile(file_pointer + ".h5"):
        raise RuntimeError("No valid h5 file found at '{0}'!"
                           .format(file_pointer + ".h5"))
    with h5py.File(file_pointer + ".h5", 'r') as hf:
        if "y" in hf:
            y = hf["y"][:]
        else:
            y = None
        for i, op in enumerate(operation_list):
            if needs_process[i]:
                X.append(process_dict[op.split(":")[0]](hf[op][:],
                         operation_details[i]))
            else:
                X.append(hf[op][:])

    logger.info("----- LOADING FROM CACHE -----")
    return X, y
