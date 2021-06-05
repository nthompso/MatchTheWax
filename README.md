# MatchTheWax
Music identifier and visualizer for vinyl record collection written in Python.

## Introduction
The enclosed code base can be deployed using a USB microphone and a Raspberry Pi 4 to identify a song and match it to an album within my personal collection of vinyl records. The song and album info is then displayed on my television using a simple splash page served using flask. Currently, the database can recognize any album in my collection which is tracked via my discogs here: https://www.discogs.com/user/nthompso13/collection. 

Due to licensing, all music was recorded raw from my turntable, and exported as mono wav files. These songs were then converted to a constellation map and imported into a database which is used to compare against the 6s snippet recorded live from the USB microphone. The current implementation is able to recognize a song on average within a database of 110+ albums within 3s on a Raspberry Pi 4. 
 
![alt text][reference]

[reference]: 

# TODO

Programming:
- [x] Import entire collection to database
- [x] Develop reliable naive identifier
- [x] Create simple splash page
- [x] Deploy code base on Raspberry Pi 4
- [ ] Add song and album info to database
  - [x] Release Year
  - [x] Genre
  - [x] LP Num
  - [x] RPM
  - [x] Track Length
  - [ ] Snackable Facts
- [x] Create visualizer

Enclosure Design:
- [ ] Design enclosure
  - [x] Raspberry Pi Mount
  - [x] Base
  - [ ] Enclosure    
- [ ] Incoporate Ring LED
- [ ] Quote and machine enclosure
