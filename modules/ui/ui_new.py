import os
import webbrowser
from typing import Tuple, Callable

import customtkinter as ctk
import cv2
from PIL import Image, ImageOps
from idlelib.tooltip import Hovertip

import modules.capturer
import modules.core
import modules.face_analyser
import modules.utilities
import modules.variables.metadata as metadata
import modules.variables.values as values

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(ctk.CTk):
    ROOT_HEIGHT = 700
    ROOT_WIDTH = 900
    PREVIEW_MAX_HEIGHT = 700
    PREVIEW_MAX_WIDTH = 1200
    RECENT_DIRECTORY_SOURCE = None
    RECENT_DIRECTORY_TARGET = None
    RECENT_DIRECTORY_OUTPUT = None
    file_types = [
        ('Image', ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')),
        ('Video', ('*.mp4', '*.mkv'))
    ]
    col01_x = 0.0625
    col02_x = 0.375
    col03_x = 0.6875

    def __init__(self, start: Callable[[], None], debug: Callable[[], None]):
        super().__init__()
        ctk.deactivate_automatic_dpi_awareness()
        ctk.set_appearance_mode('system')

        self.minsize(self.ROOT_WIDTH, self.ROOT_HEIGHT)
        self.title(f'{metadata.name} {metadata.version} {metadata.edition}')
        self.configure()
        self.protocol('WM_DELETE_WINDOW', lambda: modules.core.destroy())

        self.target_label = ctk.CTkLabel(self,
                                         text="")
        self.target_label.place(relx=self.col01_x,
                                rely=0.05,
                                relwidth=0.25,
                                relheight=0.25)

        arrow_label = ctk.CTkLabel(self, text="|", anchor="w")
        arrow_label.place(relx=0.33,
                          rely=0.10,
                          relwidth=0.05,
                          relheight=0.05)
        arrow_label = ctk.CTkLabel(self, text="|", anchor="w")
        arrow_label.place(relx=0.33,
                          rely=0.15,
                          relwidth=0.05,
                          relheight=0.05)
        arrow_label = ctk.CTkLabel(self, text="|", anchor="w")
        arrow_label.place(relx=0.33,
                          rely=0.20,
                          relwidth=0.05,
                          relheight=0.05)

        self.subject_label = ctk.CTkLabel(self,
                                          text="")
        self.subject_label.place(relx=self.col02_x,
                                 rely=0.05,
                                 relwidth=0.25,
                                 relheight=0.25)

        arrow_label = ctk.CTkLabel(self, text="➡", anchor="w")
        arrow_label.place(relx=0.65,
                          rely=0.10,
                          relwidth=0.05,
                          relheight=0.05)
        arrow_label = ctk.CTkLabel(self, text="➡", anchor="w")
        arrow_label.place(relx=0.65,
                          rely=0.15,
                          relwidth=0.05,
                          relheight=0.05)
        arrow_label = ctk.CTkLabel(self, text="➡", anchor="w")
        arrow_label.place(relx=0.65,
                          rely=0.20,
                          relwidth=0.05,
                          relheight=0.05)

        self.source_label = ctk.CTkLabel(self,
                                         text="")
        self.source_label.place(relx=self.col03_x,
                                rely=0.05,
                                relwidth=0.25,
                                relheight=0.25)

        self.target_button = ctk.CTkButton(self,
                                           text='Select the media to faceswap',
                                           cursor='hand2',
                                           command=lambda: self.select_target_path())
        self.target_button.place(relx=self.col01_x,
                                 rely=0.31,
                                 relwidth=0.25,
                                 relheight=0.1)

        self.subject_button = ctk.CTkButton(self,
                                            text='Select the face to change',
                                            cursor='hand2',
                                            command=lambda: self.select_subject_path())
        self.subject_button.place(relx=self.col02_x,
                                  rely=0.31,
                                  relwidth=0.25,
                                  relheight=0.1)

        self.source_button = ctk.CTkButton(self,
                                           text='Select the face to add',
                                           cursor='hand2',
                                           command=lambda: self.select_source_path())
        self.source_button.place(relx=self.col03_x,
                                 rely=0.31,
                                 relwidth=0.25,
                                 relheight=0.1)

        self.keep_frames_value = ctk.BooleanVar(value=values.keep_frames)
        self.keep_frames_switch = ctk.CTkSwitch(self,
                                                text='Keep frames after treatment',
                                                variable=self.keep_frames_value,
                                                cursor='hand2',
                                                command=lambda: setattr(values, 'keep_frames', self.keep_frames_value.get()))
        self.keep_frames_switch.place(relx=self.col01_x,
                                      rely=0.45,
                                      relwidth=0.25,
                                      relheight=0.05)
        keep_frames_tip = Hovertip(self.keep_frames_switch,
                                   """Keep or delete the frames, for debugging purposes.""",
                                   hover_delay=500)

        self.faces_label = ctk.CTkLabel(self, text="Faces swap:", anchor="w")
        self.faces_label.place(relx=self.col02_x, rely=0.41, relwidth=0.25, relheight=0.05)
        self.faces_value = ctk.StringVar(value=values.face_option)
        self.faces_cbox = ctk.CTkComboBox(self,
                                          values=values.faces_options,
                                          command=self.faces_callback,
                                          variable=self.faces_value)
        self.faces_value.set(values.face_option)
        self.faces_cbox.place(relx=self.col02_x, rely=0.46, relwidth=0.25, relheight=0.05)
        faces_label_tip = Hovertip(self.faces_label,
                                   """Faceswap options:
        Best one: faceswap only the face under the minimum score.
        All: faceswap all found faces.""",
                                   hover_delay=500)
        faces_cbox_tip = Hovertip(self.faces_cbox,
                                  """Faceswap options:
        Best one: faceswap only the face under the minimum score.
        All: faceswap all found faces.""",
                                  hover_delay=500)

        self.enhancer_label = ctk.CTkLabel(self, text="Enhancer mode:", anchor="w")
        self.enhancer_label.place(relx=self.col03_x, rely=0.41, relwidth=0.25, relheight=0.05)
        self.enhancer_value = ctk.StringVar(value=values.enhancer_option)
        self.enhancer_cbox = ctk.CTkComboBox(self,
                                             values=values.enhancer_options,
                                             command=self.enhancer_callback,
                                             variable=self.enhancer_value)
        self.enhancer_value.set(values.enhancer_option)
        self.enhancer_cbox.place(relx=self.col03_x, rely=0.46, relwidth=0.25, relheight=0.05)
        enhancer_label_tip = Hovertip(self.enhancer_label,
                                      """Enhancer options:
        None: No face enhancer.
        Best face only: Enhance only the face under the minimum score. May miss some frames and leave little artefacts. Can be long.
        Faces only: Enhance only the faces. May leave little artefacts. Can be very long.
        All: Analyze all frame. Very very long !""",
                                      hover_delay=500)
        enhancer_cbox_tip = Hovertip(self.enhancer_cbox,
                                     """Enhancer options:
        None: No face enhancer.
        Best face only: Enhance only the face under the minimum score. May miss some frames and leave little artefacts. Can be long.
        Faces only: Enhance only the faces. May leave little artefacts. Can be very long.
        All: Analyze all frame. Very very long !""",
                                     hover_delay=500)

        nsfw_value = ctk.BooleanVar(value=values.nsfw)
        nsfw_switch = ctk.CTkSwitch(self,
                                    text='Content is NSFW',
                                    variable=nsfw_value,
                                    cursor='hand2',
                                    command=lambda: setattr(values, 'nsfw', nsfw_value.get()))
        nsfw_switch.place(relx=self.col01_x, rely=0.52, relwidth=0.25, relheight=0.05)

        self.distance_value = ctk.IntVar(value=values.distance_score)
        self.distance_label = ctk.CTkLabel(self, text="Minimum score limit :", anchor="w")
        self.distance_label.place(relx=self.col02_x, rely=0.51, relwidth=0.25, relheight=0.05)
        self.distance = ctk.CTkSlider(self, from_=0, to=40, number_of_steps=40, command=self.slider_callback, variable=self.distance_value)
        self.distance.place(relx=self.col02_x, rely=0.56, relwidth=0.50, relheight=0.03)
        self.distance_score = ctk.CTkLabel(self, text=str(self.distance_value.get()), anchor="w")
        self.distance_score.place(relx=self.col02_x+0.25, rely=0.51, relwidth=0.05, relheight=0.05)
        distance_label_tip = Hovertip(self.distance_label,
                                      """Distance score. Lower is selective, higher is generous.
        10-15: Near from perfection, possible for a photo.
        15-20: Can be realistic in a short video
        20-25: Realistic for videos
        25-30: Low quality videos
        35-40: You should faceswap "All", or use another "face to change" file.""",
                                      hover_delay=500)
        distance_tip = Hovertip(self.distance,
                                """Distance score. Lower is selective, higher is generous.
        10-15: Near from perfection, possible for a photo.
        15-20: Can be realistic in a short video
        20-25: Realistic for videos
        25-30: Low quality videos
        35-40: You should faceswap "All", or use another "face to change" file.""",
                                hover_delay=500)

        self.start_button = ctk.CTkButton(self, text='< Start >', cursor='hand2', command=lambda: self.select_output_path_and_start(start))
        self.start_button.place(relx=self.col01_x, rely=0.64, relwidth=0.875, relheight=0.09)
        start_button_tip = Hovertip(self.start_button,
                                    """Select output path and start faceswap.""",
                                    hover_delay=500)

        self.preview_button = ctk.CTkButton(self, text='( Preview )', cursor='hand2', command=lambda: self.toggle_preview())
        self.preview_button.place(relx=self.col01_x, rely=0.74, relwidth=0.875, relheight=0.08)
        preview_button_tip = Hovertip(self.preview_button,
                                      """See results on some frames. Useful for videos, or to test enhance options.""",
                                      hover_delay=500)

        self.debug_button = ctk.CTkButton(self, text='[ Debug matching scores ]', cursor='hand2', command=lambda: self.select_output_path_and_debug(debug))
        self.debug_button.place(relx=self.col01_x, rely=0.83, relwidth=0.875, relheight=0.07)
        debug_button_tip = Hovertip(self.debug_button,
                                    """Generate photo/video with information about matching scores.""",
                                    hover_delay=500)

        self.status_label = ctk.CTkLabel(self, text=None, justify='center')
        self.status_label.place(relx=self.col01_x, rely=0.9, relwidth=0.875, relheight=0.05)

        self.donate_label = ctk.CTkLabel(self, text='bzerath GitHub', justify='center', cursor='hand2')
        self.donate_label.place(relx=self.col01_x, rely=0.95, relwidth=0.875, relheight=0.05)
        self.donate_label.configure(text_color=["gray74", "gray60"])
        self.donate_label.bind('<Button>', lambda event: webbrowser.open('https://github.com/bzerath/ReActor-UI'))

        self.PREVIEW = self.create_preview()

    def faces_callback(self, choice):
        values.face_option = self.faces_value.get()

    def enhancer_callback(self, choice):
        values.enhancer_option = self.enhancer_value.get()
        if values.enhancer_option != values.enhancer_none:
            values.fp_ui['face_enhancer'] = True
        else:
            values.fp_ui['face_enhancer'] = False

    def slider_callback(self, _):
        self.distance_score.configure(text=str(self.distance_value.get()))
        values.distance_score = int(self.distance_value.get())

    def select_source_path(self) -> None:
        source_path = ctk.filedialog.askopenfilename(title='select an source image',
                                                     initialdir=self.RECENT_DIRECTORY_SOURCE,
                                                     filetypes=[self.file_types[0]])
        # self.PREVIEW.withdraw()
        if modules.utilities.is_image(source_path):
            values.source_path = source_path
            print("values.source_path", values.source_path)
            self.RECENT_DIRECTORY_SOURCE = os.path.dirname(values.source_path)
            image = self.render_image_preview(values.source_path,
                                              (200, 200))
            print(image)
            self.source_label.configure(image=image)
        else:
            values.source_path = None
            self.source_label.configure(image=None)

    def select_target_path(self) -> None:
        target_path = ctk.filedialog.askopenfilename(title='select an target image or video',
                                                     initialdir=self.RECENT_DIRECTORY_TARGET,
                                                     filetypes=self.file_types)
        if modules.utilities.is_image(target_path):
            values.target_path = target_path
            print("values.target_path", values.target_path)
            self.RECENT_DIRECTORY_TARGET = os.path.dirname(values.target_path)
            image = self.render_image_preview(values.target_path,
                                              (200, 200))
            self.target_label.configure(image=image)
        elif modules.utilities.is_video(target_path):
            values.target_path = target_path
            self.RECENT_DIRECTORY_TARGET = os.path.dirname(values.target_path)
            video_frame = self.render_video_preview(target_path, (200, 200))
            self.target_label.configure(image=video_frame)
        else:
            values.target_path = None
            self.target_label.configure(image=None)

    def select_subject_path(self) -> None:
        subject_path = ctk.filedialog.askopenfilename(title='select a subject image',
                                                      initialdir=self.RECENT_DIRECTORY_SOURCE,
                                                      filetypes=[self.file_types[0]])
        if modules.utilities.is_image(subject_path):
            values.subject_path = subject_path
            print("values.subject_path", values.subject_path)
            self.RECENT_DIRECTORY_SOURCE = os.path.dirname(values.subject_path)
            image = self.render_image_preview(values.subject_path,
                                              (200, 200))
            self.subject_label.configure(image=image)
        else:
            values.subject_path = None
            self.subject_label.configure(image=None)

    def select_output_path_and_start(self, start: Callable[[], None]) -> None:
        if modules.utilities.is_image(values.target_path):
            output_path = ctk.filedialog.asksaveasfilename(title='save image output file',
                                                           filetypes=[self.file_types[0]],
                                                           defaultextension='.png',
                                                           initialfile=os.path.splitext(os.path.basename(values.target_path))[0]+"_output.png"
                                                           if values.target_path else 'output.png',
                                                           initialdir=self.RECENT_DIRECTORY_OUTPUT)
        elif modules.utilities.is_video(values.target_path):
            output_path = ctk.filedialog.asksaveasfilename(title='save video output file',
                                                           filetypes=[self.file_types[1]],
                                                           defaultextension='.mp4',
                                                           initialfile=os.path.splitext(os.path.basename(values.target_path))[0]+"_output.mp4"
                                                           if values.target_path else 'output.mp4',
                                                           initialdir=self.RECENT_DIRECTORY_OUTPUT)
        else:
            output_path = None
        if output_path:
            values.output_path = output_path
            self.RECENT_DIRECTORY_OUTPUT = os.path.dirname(values.output_path)
            print(self.infos())
            start()

    def select_output_path_and_debug(self, debug: Callable[[], None]) -> None:
        if modules.utilities.is_image(values.target_path):
            output_path = ctk.filedialog.asksaveasfilename(title='save image output file',
                                                           filetypes=[self.file_types[0]],
                                                           defaultextension='.png',
                                                           initialfile=os.path.splitext(os.path.basename(values.target_path))[0]+"_debug.png"
                                                           if values.target_path else 'debug.png',
                                                           initialdir=self.RECENT_DIRECTORY_OUTPUT)
        elif modules.utilities.is_video(values.target_path):
            output_path = ctk.filedialog.asksaveasfilename(title='save video output file',
                                                           filetypes=[self.file_types[1]],
                                                           defaultextension='.mp4',
                                                           initialfile=os.path.splitext(os.path.basename(values.target_path))[0]+"_debug.mp4"
                                                           if values.target_path else 'debug.mp4',
                                                           initialdir=self.RECENT_DIRECTORY_OUTPUT)
        else:
            output_path = None
        if output_path:
            values.output_path = output_path
            self.RECENT_DIRECTORY_OUTPUT = os.path.dirname(values.output_path)
            print(self.infos())
            debug()

    def create_preview(self) -> ctk.CTkToplevel:
        preview = ctk.CTkToplevel(self)
        preview.withdraw()
        preview.title('Preview')
        preview.configure()
        preview.protocol('WM_DELETE_WINDOW', lambda: self.toggle_preview())
        preview.resizable(width=False, height=False)

        self.preview_label = ctk.CTkLabel(preview, text=None)
        self.preview_label.pack(fill='both', expand=True)

        self.preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: self.update_preview(frame_value))

        return preview

    def update_preview(self, frame_number: int = 0) -> None:
        if values.source_path and values.target_path:
            temp_frame = modules.capturer.get_video_frame(values.target_path, frame_number)
            if not values.nsfw:
                from modules.predicter import predict_frame
                if predict_frame(temp_frame):
                    quit()
            for frame_processor in modules.core.get_frame_processors_modules(values.frame_processors):
                subject_face = modules.face_analyser.get_face_analyser().get(cv2.imread(values.subject_path))
                if subject_face:
                    subject_embedding = subject_face[0].embedding
                else:
                    raise Exception("Subject face does not contain face...")
                temp_frame = frame_processor.process_frame(
                    modules.face_analyser.get_one_face(cv2.imread(values.source_path)),
                    temp_frame,
                    subject_embedding
                )
            image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(image, (self.PREVIEW_MAX_WIDTH, self.PREVIEW_MAX_HEIGHT), Image.LANCZOS)
            image = ctk.CTkImage(image, size=image.size)
            self.preview_label.configure(image=image)

    def init_preview(self) -> None:
        if modules.utilities.is_image(values.target_path):
            self.preview_slider.pack_forget()
        if modules.utilities.is_video(values.target_path):
            video_frame_total = modules.capturer.get_video_frame_total(values.target_path)
            self.preview_slider.configure(to=video_frame_total)
            self.preview_slider.pack(fill='x')
            self.preview_slider.set(0)

    def toggle_preview(self) -> None:
        if self.PREVIEW.state() == 'normal':
            self.PREVIEW.withdraw()
        elif values.source_path and values.target_path:
            self.init_preview()
            self.update_preview()
            self.PREVIEW.deiconify()

    @staticmethod
    def render_image_preview(image_path: str,
                             size: Tuple[int, int]) -> ctk.CTkImage:
        image = Image.open(image_path)
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)

    @staticmethod
    def render_video_preview(video_path: str,
                             size: Tuple[int, int],
                             frame_number: int = 0) -> ctk.CTkImage:
        capture = cv2.VideoCapture(video_path)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.LANCZOS)
            capture.release()
            cv2.destroyAllWindows()
            return ctk.CTkImage(image, size=image.size)
        capture.release()
        cv2.destroyAllWindows()

    def infos(self):
        return f"""
        Faceswapping {values.target_path} to {values.output_path}.
        Will replace {values.subject_path} face by {values.source_path} face.
        Options:
            keep frames : {values.keep_frames}
            enhancer : {values.enhancer_option}
            faceswap : {values.face_option}
            distance : {values.distance_score}
            nsfw : {values.nsfw}
        """