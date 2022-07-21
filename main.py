import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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
ZOOM_WIDTH = FRAME_WIDTH // 3
ZOOM_HEIGHT = FRAME_HEIGHT // 3


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


def edit_annotation(df, index):
    layout_id = [
        [
            sg.Text('Tracking ID', size=(8, 1)),
            sg.InputText(df.iloc[index, 1], size=(4, 1), key='new_id')
        ],
        [
            sg.Checkbox('Edit this frame only', False, key='id_only_frame')
        ]
    ]
    layout_bbox = [
        [
            sg.Text('Left', size=(6, 1)),
            sg.InputText(df.iloc[index, 2], size=(5, 1), key='new_bbox_left')
        ],
        [
            sg.Text('Top', size=(6, 1)),
            sg.InputText(df.iloc[index, 3], size=(5, 1), key='new_bbox_top')
        ],
        [
            sg.Text('Width', size=(6, 1)),
            sg.InputText(df.iloc[index, 4], size=(5, 1), key='new_bbox_width')
        ],
        [
            sg.Text('Height', size=(6, 1)),
            sg.InputText(df.iloc[index, 5], size=(5, 1), key='new_bbox_height')
        ]
    ]
    layout_class = [
        [
            sg.Text('Class ID', size=(10, 1)),
            sg.InputText(df.iloc[index, 6], size=(4, 1), key='new_class_id')
        ],
        [
            sg.Text('Sub Class ID', size=(10, 1)),
            sg.InputText(df.iloc[index, 8], size=(4, 1), key='new_subclass_id')
        ],
        [
            sg.Checkbox('Edit this frame only', False, key='class_only_frame')
        ]
    ]
    layout_conf = [[sg.InputText(df.iloc[index, 7], size=(11, 1), key='new_conf')]]

    fra_id = sg.Frame(layout=layout_id, title='Tracking', vertical_alignment='top')
    fra_bbox = sg.Frame(layout=layout_bbox, title='BBox', vertical_alignment='top')
    fra_class = sg.Frame(layout=layout_class, title='Class', vertical_alignment='top')
    fra_conf = sg.Frame(layout=layout_conf, title='Confidence', vertical_alignment='top')
    # fra_id = sg.Column(layout=layout_id, vertical_alignment='top')
    # fra_bbox = sg.Column(layout=layout_bbox, vertical_alignment='top')
    
    layout = [
        [fra_id, fra_bbox, fra_class, fra_conf],
        [sg.Submit('submit'), sg.Cancel('cancel')]
    ]

    window = sg.Window(f'Editing', layout)
    window.finalize()

    while True:
        event, values = window.read(timeout=None)
        if event == 'cancel' or sg.WIN_CLOSED:
            id = df.iloc[index, 1]
            bbox_left = df.iloc[index, 2]
            bbox_top = df.iloc[index, 3]
            bbox_width = df.iloc[index, 4]
            bbox_height = df.iloc[index, 5]
            class_id = df.iloc[index, 6]
            subclass_id = df.iloc[index, 8]
            conf = df.iloc[index, 7]
            break
        elif event == 'submit':
            print('\nEdit annotations:')
            id          = values['new_id'];          print(df.iloc[index, 1], '=>', id)
            bbox_left   = values['new_bbox_left'];   print(df.iloc[index, 2], '=>', bbox_left)
            bbox_top    = values['new_bbox_top'];    print(df.iloc[index, 3], '=>', bbox_top)
            bbox_width  = values['new_bbox_width'];  print(df.iloc[index, 4], '=>', bbox_width)
            bbox_height = values['new_bbox_height']; print(df.iloc[index, 5], '=>', bbox_height)
            class_id    = values['new_class_id'];    print(df.iloc[index, 6], '=>', class_id)
            subclass_id = values['new_subclass_id']; print(df.iloc[index, 8], '=>', subclass_id)
            conf        = values['new_conf'];        print(df.iloc[index, 7], '=>', conf)
            break

    window.close()

    return TrackingData(id, bbox_left, bbox_top, bbox_width, bbox_height, class_id, subclass_id, conf)


def read_tracks(f_txt, ftype='txt'):
    # imp csv read
    names = [
        'frame_id', 'id', 'bbox_left', 'bbox_top', 'bbox_width',
        'bbox_height', 'class_id', 'conf'
    ]
    usecols = [0, 1, 2, 3, 4, 5, 10, 11]
    df = pd.read_csv(f_txt, sep=' ', names=names, usecols=usecols)

    # add subclass_id
    df['subclass_id'] = 0

    return df


def draw_annotations(img, df, cmap, values):
    for i, row in df.iterrows():        
        pt1 = int(row.bbox_left), int(row.bbox_top)
        pt2 = int(row.bbox_left + row.bbox_width), int(row.bbox_top + row.bbox_height)
        clsid = int(row.class_id)
        cls = CLASSES[clsid]
        id = int(row.id)
        conf = row.conf
        org = int(row.bbox_left), int(row.bbox_top) - 10

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


def update_annotations(df, window, image, cmap, values):
    df_ = df.copy()
    if df_.empty:
        window['anno_table'].update('')
    else:
        draw_annotations(image, df_, cmap, values)
        df_['conf'] = df_['conf'].apply(lambda x: int(x*100))
        window['anno_table'].update(df_.to_numpy().tolist())

    return df_

        
def layout_ids(ids, cols=29):
    ret = []
    for i, id in enumerate(ids):
        if i % cols == 0:
            tmp = []
        elif i % cols == cols-1:
            ret.append(tmp)
        tmp.append(sg.Checkbox(f'{id:03d}', True, key=f'-{id}-'))
    ret.append(tmp)

    return ret        


class TrackingData:
    def __init__(self, id, bbox_left, bbox_top, bbox_width, bbox_height, class_id, subclass_id, conf):
        self.id = int(id)
        self.bbox_left = int(bbox_left)
        self.bbox_top = int(bbox_top)
        self.bbox_width = int(bbox_width)
        self.bbox_height = int(bbox_height)
        self.class_id = int(class_id)
        self.subclass_id = int(subclass_id)
        self.conf = float(conf)

        
class TrackingDataEditingTools:
    def __init__(self):
        # self.fp, self.file_txt = load_video_and_tracking()
        self.fp = Path('samples/A000.mp4')
        self.file_txt = Path('samples/A000.txt')
        # self.fp = Path('samples/D005.mp4')
        # self.file_txt = Path('samples/D005.txt')
        
        if not self.fp.exists() or not self.file_txt.exists():
            sg.Popup('File not found.')
            exit()

        # csv output
        self.file_csv = Path(self.file_txt.stem + '.csv')

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

            self.zoom_left = self.width // 2
            self.zoom_top = self.height // 2
        else:
            sg.Popup('Failed to load video.')
            exit()

        # tracking data
        self.track = read_tracks(str(self.file_txt))
        self.track_ids = set(self.track['id'])
        self.num_tracks = len(self.track_ids)
        self.df = self.track[self.track['frame_id'] == self.frame_id]


        # colormap
        self.cmap = plt.get_cmap(COLORMAP, len(CLASSES))

        # layout
        layout_zoom = [[sg.Image(filename='', key='sub_screen')]]
        layout_file = [
            [
                sg.Text('Video', size=(7, 1)),
                sg.InputText(os.path.abspath(str(self.fp)), size=(70, 1)),
            ],
            [
                sg.Text('Tracking', size=(7, 1)),
                sg.InputText(os.path.abspath(str(self.file_txt)), size=(70, 1)),
            ]
        ]
        layout_ctrl = [
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
                    size=(33.5, 15),
                    key='-progress bar-',
                    enable_events=True
                ),
                sg.Text(self.frame_end),
            ]
        ]
        layout_cls = [
            [
                sg.Checkbox(v, True, key=f'-{v}-') for k, v in CLASSES.items()
            ]
        ]
        layout_mode = [
            [
                sg.Radio('detection', key='key_radio_detection', group_id='g_mode', default=True),
                sg.Radio('tracking', key='key_radio_tracking', group_id='g_mode', default=False)
            ]
        ]
        layout_track = layout_ids(self.track_ids)

        self.screen = sg.Image(filename='', key='screen')
        self.anno_table = sg.Table(
            self.df.to_numpy().tolist(),
            headings=self.df.columns.tolist(),
            def_col_width=9,
            auto_size_columns=False,
            hide_vertical_scroll=True,
            num_rows=25,
            key='anno_table',
            enable_events=True
        )

        # Frames
        self.fra_zoom = sg.Frame(layout=layout_zoom, title='', vertical_alignment='top')
        self.fra_file = sg.Frame(layout=layout_file, title='', vertical_alignment='top')
        self.fra_ctrl = sg.Frame(layout=layout_ctrl, title='', vertical_alignment='top')
        self.fra_clss = sg.Frame(layout=layout_cls,  title='Classes', vertical_alignment='top')
        self.fra_colr = sg.Frame(layout=layout_mode, title='BBox Coloring', vertical_alignment='top')
        self.fra_trak = sg.Frame(layout=layout_track, title=f'Tracking IDs ({len(self.track_ids)})', vertical_alignment='top')

        # Column
        self.layout_col00 = sg.Column(
            [
                [self.screen],
                [self.fra_trak]
            ],
            vertical_alignment='top'
        )
        self.layout_col10 = sg.Column(
            [
                [self.fra_zoom],
                [self.fra_file],
                [self.fra_ctrl],
                [self.fra_clss],
                [self.fra_colr],
                [self.anno_table],
                [sg.Button('Quit')]
            ],
            vertical_alignment='top'
        )

    def run(self):
        layout = [[self.layout_col00, self.layout_col10]]

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

                # update annotations
                # self.df = self.track[self.track['frame_id'] == self.frame_id]
                # update_annotations(self.df, window, self.frame, self.cmap, values)

                # events
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

                # capture frame
                self.ret, self.frame = self.cap.read()
                self.valid_frame = int(self.num_frames - self.frame_begin)

                if not self.ret:
                    # self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_begin)
                    # self.num_frames = self.frame_begin
                    continue

                # update annotations
                self.df = self.track[self.track['frame_id'] == self.frame_id]
                update_annotations(self.df, window, self.frame, self.cmap, values)

                if self.event == 'anno_table':
                    # idx = values['anno_table'][0]
                    td = edit_annotation(self.df, *values['anno_table'])
                    update_annotations(self.df, window, self.frame, self.cmap, values)
                    self.zoom_left = td.bbox_left - ZOOM_WIDTH // 2
                    self.zoom_left = 0 if self.zoom_left < 0 else self.zoom_left
                    self.zoom_top = td.bbox_top - ZOOM_HEIGHT //2
                    self.zoom_top = 0 if self.zoom_top < 0 else self.zoom_top
                    
                
                # update display
                # main screen
                _, data = cv2.imencode(
                    '.png',
                    cv2.resize(self.frame, dsize=(FRAME_WIDTH, FRAME_HEIGHT)),
                )
                window['screen'].update(data=data.tobytes())
                # sub screen
                _, data1 = cv2.imencode(
                    '.png',
                    self.frame[self.zoom_top:self.zoom_top+ZOOM_HEIGHT,
                               self.zoom_left:self.zoom_left+ZOOM_WIDTH]
                )
                window['sub_screen'].update(data=data1.tobytes())

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
    
