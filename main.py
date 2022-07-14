import cv2
import numpy as np
import matplotlib.pyplot as plt
# import mycolorpy.colorlist as mcp
import pandas as pd
import PySimpleGUI as sg
import time
from pathlib import Path

CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car'
}
COLORMAP = 'gist_rainbow'
NUM_COLOR = 10
FONTSTYLE = cv2.FONT_HERSHEY_SIMPLEX
FONTSIZE = 1.2
FRAME_WIDTH = 3840 // 2
FRAME_HEIGHT = 2160 // 2


def load_video_and_tracking():
    fp = ''
    f_track = ''

    layout = [
        [
            sg.Text('Video File', size=(10, 1)),
            sg.InputText(size=(70, 10)),
            sg.FileBrowse(key='video_file'),
        ],
        [
            sg.Text('Tracking File', size=(10, 1)),
            sg.InputText(size=(70, 10)),
            sg.FileBrowse(key='tracking_file'),
        ],
        [
            sg.Submit(key='submit'),
            sg.Cancel('Exit')
        ]
    ]

    window = sg.Window('Select file', layout)
    window.finalize()

    while True:
        event, values = window.read(timeout=None)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        elif event == 'submit':
            if values[0] == '':
                sg.popup('File not found.')
                event = ''
            else:
                fp = values[0]
                f_track = values[1]
                break

    window.close()

    return Path(fp), Path(f_track)


def read_tracks(f_txt):
    names = [
        'frame_id', 'id', 'bbox_left', 'bbox_top', 'bbox_width',
        'bbox_height', 'class_id', 'conf'
    ]
    usecols = [0, 1, 2, 3, 4, 5, 10, 11]
    return pd.read_csv(f_txt, sep=' ', names=names, usecols=usecols)


def draw_annotations(img, df, cmap, values):
    for i, row in df.iterrows():        
        pt1 = int(row.bbox_left), int(row.bbox_top)
        pt2 = int(row.bbox_left + row.bbox_width), int(row.bbox_top + row.bbox_height)
        clsid = int(row.class_id)
        cls = CLASSES[clsid]
        id = int(row.id)
        conf = row.conf
        org = int(row.bbox_left), int(row.bbox_top)-10

        if values['key_radio_detection']:
            color = [int(_*255) for _ in cmap(clsid % len(CLASSES))]
            color[0], color[2] = color[2], color[0]  # RGB -> BGR
            txt = f'{cls} {conf:.0%}'
        else:
            color = [int(_*255) for _ in cmap(id % NUM_COLOR)]
            color[0], color[2] = color[2], color[0]  # RGB -> BGR
            txt = f'{id:03d}'

        if values[f'-{cls}-'] and values[f'-{id}-']:
            cv2.rectangle(img, pt1, pt2, color, thickness=2)
            cv2.putText(img, txt, org, FONTSTYLE, FONTSIZE, color, thickness=2)


def print_annotations(df, window, image, cmap, values):
    if df.empty:
        window['-table-'].update('')
    else:
        draw_annotations(image, df, cmap, values)
        window['-table-'].update(df.to_numpy().tolist())

        
def layout_ids(ids):
    ret = []
    for i, id in enumerate(ids):
        if i % 10 == 0:
            tmp = []
        elif i % 10 == 9:
            ret.append(tmp)
        tmp.append(sg.Checkbox(f'{id:03d}', True, key=f'-{id}-'))
    ret.append(tmp)

    return ret        


class TrackingDataEditingTools:
    def __init__(self):
        self.fp, self.f_track = load_video_and_tracking()
        # self.fp = Path('samples/A000.mp4')
        # self.f_track = Path('samples/A000.txt')
        
        if not len(self.fp.name) or not len(self.f_track.name):
            sg.Popup('File not found.')
            exit()

        # video capture
        self.cap = cv2.VideoCapture(str(self.fp))
        if not self.cap.isOpened():
            sg.Popup('Failed to open video.')
            exit()
            
        self.ret, self.frame = self.cap.read()
        if self.ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.frame_id = 0
            self.frame_begin = 0
            self.frame_end = self.num_frames - 1

            self.is_stop = True
        else:
            sg.Popup('Failed to load video.')
            exit()

        # tracking data
        self.track = read_tracks(str(self.f_track))
        self.track_ids = set(self.track['id'])
        self.num_tracks = len(self.track_ids)
        self.df = self.track[self.track['frame_id'] == self.frame_id]


        # colormap
        self.cmap = plt.get_cmap(COLORMAP, len(CLASSES))

        # layout
        layout_disp = [[sg.Image(filename='', key='-display-')]]
        layout_ctrl = [
            [
                sg.Text('Video File', size=(10, 1)),
                sg.InputText(self.fp.name, size=(70, 1)),
                # sg.FileBrowse(key='video_file'),
            ],
            [
                sg.Text('Tracking File', size=(10, 1)),
                sg.InputText(self.f_track.name, size=(70, 10)),
                # sg.FileBrowse(key='tracking_file'),
            ],
            # [
            #     sg.Button('Load')
            # ],
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Button('|<'),
                sg.Button('<'),
                sg.Button('Play / Stop'),
                sg.Button('>'),
                sg.Button('>|'),
                sg.Text(self.frame_begin),
                sg.Slider(
                    (0, self.frame_end),
                    0,
                    1,
                    orientation='h',
                    size=(35, 15),
                    key='-progress bar-',
                    enable_events=True
                ),
                sg.Text(self.frame_end),
            ],
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Text('Classes')
            ],
            [
                sg.Checkbox(v, True, key=f'-{v}-') for k, v in CLASSES.items()
            ],
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Text('BBox Coloring')
            ],
            [
                sg.Radio('detection', key='key_radio_detection', group_id='g_mode', default=True),
                sg.Radio('tracking', key='key_radio_tracking', group_id='g_mode', default=False)
            ],
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Text('Tracking ID')
            ],
            *layout_ids(self.track_ids),
            [
                sg.HorizontalSeparator()
            ],
            [
                sg.Table(
                    self.df.to_numpy().tolist(),
                    headings=self.df.columns.tolist(),
                    auto_size_columns=False,
                    vertical_scroll_only=False,
                    num_rows=16,
                    key='-table-'
                )
            ],
            [
                sg.Button('Quit')
            ]
        ]

        self.disp = sg.Frame(layout=layout_disp, title='', relief=sg.RELIEF_SUNKEN)
        self.ctrl = sg.Frame(layout=layout_ctrl, title='', relief=sg.RELIEF_SUNKEN)

    def run(self):
        layout = [[self.disp, self.ctrl]]
        window = sg.Window('Tracking Data Editing Tools', layout, location=(0, 0))
        window.finalize()
        self.event, values = window.read(timeout=0)

        print('Load video')
        print(f'Path: {self.fp}')
        print(f'FPS: {self.fps: .0f}')
        print(f'Width: {self.width: .0f}')
        print(f'Height: {self.height: .0f}')
        print(f'Last frame: {self.num_frames}')
        print(f'Tracking #: {self.num_tracks}')

        # main loop
        try:
            while True:
                self.event, values = window.read(timeout=0)
                
                if self.event == 'Quit':
                    break

                if self.event != '__TIMEOUT__':
                    print(self.event)

                if self.event in ('Exit', sg.WIN_CLOSED, None):
                    break

                if self.event == 'Play / Stop':
                    self.is_stop = not self.is_stop

                if self.event == '>|':
                    self.frame_id = self.frame_end
                    window['-progress bar-'].update(self.frame_id)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

                if self.event == '>':
                    self.frame_id += 1
                    window['-progress bar-'].update(self.frame_id)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

                if self.event == '<':
                    self.frame_id = np.maximum(0, self.frame_id - 1)
                    window['-progress bar-'].update(self.frame_id)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

                if self.event == '|<':
                    self.frame_id = 0
                    window['-progress bar-'].update(self.frame_id)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

                if self.event == '-progress bar-':
                    self.frame_id = int(values['-progress bar-'])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

                if values['key_radio_detection']:
                    self.cmap = plt.get_cmap(COLORMAP, len(CLASSES))
                else:
                    self.cmap = plt.get_cmap(COLORMAP, NUM_COLOR)

                window['-progress bar-'].update(self.frame_id)
                # if self.is_stop and self.event == '__TIMEOUT__':
                #     window['-progress bar-'].update(self.frame_id)
                #     continue

                # capture frame
                self.ret, self.frame = self.cap.read()
                self.valid_frame = int(self.num_frames - self.frame_begin)

                if not self.ret:
                    # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_begin)
                    # self.num_frames = self.frame_begin
                    continue
                
                self.df = self.track[self.track['frame_id'] == self.frame_id]
                print_annotations(self.df, window, self.frame, self.cmap, values)
                
                # update display
                _, data = cv2.imencode(
                    '.png',
                    cv2.resize(self.frame, dsize=(FRAME_WIDTH, FRAME_HEIGHT)),
                )
                window['-display-'].update(data=data.tobytes())

                if self.is_stop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
                else:
                    self.frame_id += 1
                    window['-progress bar-'].update(self.frame_id + 1)

        finally:
            self.cap.release()
            window.close()


if __name__ == '__main__':
    tdet = TrackingDataEditingTools()
    tdet.run()
    
