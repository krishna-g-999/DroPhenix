# FreeClimber example data — attribution and licence

The files in this directory are **not DroPhenix's work**. They are the example
recording and the example outputs distributed with FreeClimber, redistributed
here under its MIT licence, whose terms require this notice to travel with
them.

## What is here, and why

| File | What it is |
|---|---|
| `w1118_m_2_1.h264` | the example recording, as the H.264 elementary stream FreeClimber's own config processes (`file_suffix="h264"`) |
| `w1118_m_2_1.mp4` | the same recording re-encoded; kept because decoding it instead measurably shifts the detections, which the suite records |
| `example.cfg` | the analysis parameters FreeClimber used |
| `w1118_m_2_1.raw.csv` | FreeClimber's own 4,289 located spots, with the `True_particle` column its filters produced |
| `w1118_m_2_1.filtered.csv` | its 4,093 surviving spots |
| `w1118_m_2_1.slopes.csv`, `results.csv` | its published velocity output |

They are here so that DroPhenix's reimplementation of FreeClimber's method can
be checked against FreeClimber's OWN published answer rather than against a
careful reading of its source — see
`validation/validate_freeclimber_detection.py` and
`validation/validate_freeclimber_benchmark.py`. A reimplementation used as a
reference has to be shown faithful before it judges anything, and these files
are what make that possible.

## Citation

Spierer AN, Yoon D, Zhu C-T, Rand DM (2021) FreeClimber: automated
quantification of climbing performance in *Drosophila*. *Journal of
Experimental Biology* 224:jeb229377. doi:10.1242/jeb.229377

Source repository: https://github.com/adamspierer/FreeClimber

## Licence (MIT), reproduced in full as its terms require

```
Copyright 2020 Adam Spierer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

