# ASL Sign Detection

Real-time American Sign Language handshape detection using your Mac's webcam and the
Jetson's MediaPipe hand tracking, connected over the USB-C link.

## What it does

Your Mac captures webcam frames and sends them as JPEGs to the Jetson. The Jetson runs
MediaPipe to extract 21 hand landmarks per hand, classifies the handshape into an ASL sign,
and sends the result back as JSON. Your Mac shows the detected sign live on the video window.
All of the heavy vision work runs on the Jetson, so the Mac only needs a webcam and OpenCV.

## Architecture

```
  Mac client                                 Jetson server
  192.168.55.100                             192.168.55.1:8765
  (webcam + OpenCV)                          (MediaPipe hand landmarks + classifier)

      camera frame
      ----------- JPEG ------------------------->
                                              extract 21 landmarks per hand
                                              classify handshape -> ASL sign
      <---------- JSON (sign + confidence) ------
      draw label on the live video
```

The two devices talk over the USB-C link (Mac 192.168.55.100, Jetson 192.168.55.1),
so no Wi-Fi or internet is needed.

## How to use

1. On the Jetson, start the server. Run the menu and pick option 9 (ASL sign detection):

   ```
   menu
   ```

   Or start it directly:

   ```
   python3 /mnt/nvme/brain/asl_server.py
   ```

2. On the Mac, start the webcam client:

   ```
   cd ~/jetson-brain && ./asl-mac.sh
   ```

   A video window opens. Show a sign to the camera and the detected label appears on screen.
   Press q (or close the window) to quit.

## Built-in starter signs (no training required)

These are recognized out of the box from hand geometry alone:

- 5 / open hand
- S / fist
- 1 / point
- V / two
- L
- W / three
- I / pinky
- ILY (I love you)
- thumbs up

## Teach it a custom sign

You can add your own signs with a quick collection pass. On the Jetson:

1. Start collection for the sign you want, for example HELLO:

   ```
   python3 /mnt/nvme/brain/asl_server.py --collect HELLO
   ```

2. On the Mac, run `./asl-mac.sh` and hold the sign steady to the webcam for a few seconds
   so it captures several samples.

3. Press Ctrl-C on the Jetson to save the samples for that sign.

4. Repeat for each sign you want to add (run `--collect NEXTSIGN` again, hold, Ctrl-C).

Future normal runs automatically load the saved KNN dataset at
`/mnt/nvme/brain/asl_dataset.json` and recognize your custom signs alongside the built-in ones.

## How it works

MediaPipe returns 21 landmarks per hand. The landmarks are normalized (translation and scale
removed) so the same handshape reads the same regardless of where the hand is or how far it is
from the camera. Classification uses simple geometry rules for the built-in starter signs
(which fingers are extended, thumb position, and so on), plus an optional KNN match against your
collected dataset when custom signs have been taught.

## Troubleshooting

- If the Mac client cannot connect: is option 9 (the ASL server) actually running on the Jetson,
  and is the USB-C link up (can you reach 192.168.55.1)?
