
# This script uses the automatic estimation of min and max f0 proposed by Daniel Hirst
# Hirst, Daniel. (2007). A Praat plugin for Momel and INTSINT with improved algorithms for modelling and coding intonation. Proceedings of the 16th International Congress of Phonetic Sciences.
# https://www.researchgate.net/publication/228640428_A_Praat_plugin_for_Momel_and_INTSINT_with_improved_algorithms_for_modelling_and_coding_intonation
#
# Pitch ceiling raised from q3*1.5 to q3*2.5 to allow for expressive speech, as described at:
# "Hirst, Daniel. (2011). The analysis by synthesis of speech melody: from data to models"
# https://www.researchgate.net/publication/228409777_The_analysis_by_synthesis_of_speech_melody_from_data_to_models

# http://www.praatvocaltoolkit.com/

# Adaptiation from Schwab and Corretge

form Input directory name
    comment Enter directory where soundfile lives:
    sentence soundDir
    comment Enter name of sound file
    sentence sound
endform


name$ = sound$ - ".wav"

# new pitch file
calcPitch$ = name$ + ".Pitch"

# Reads in sound (wav) 
Read from file... 'soundDir$'/'name$'.wav

# deletes any existing output file
deleteFile$ = soundDir$ + calcPitch$
filedelete 'deleteFile$'

select Sound 'name$'

selsnd_m = selected("Sound")

nocheck noprogress To Pitch: 0, 40, 600

if extractWord$(selected$(), "") = "Pitch"
    voicedframes = Count voiced frames

    if voicedframes > 0
        q1 = Get quantile: 0, 0, 0.25, "Hertz"
        q3 = Get quantile: 0, 0, 0.75, "Hertz"
        minF0 = round(q1 * 0.75)
        maxF0 = round(q3 * 2.5)
    else
        minF0 = 40
        maxF0 = 600
    endif

    Remove
else
    minF0 = 40
    maxF0 = 600
endif

if minF0 < 3 / (object[selsnd_m].nx * object[selsnd_m].dx)
    minF0 = ceiling(3 / (object[selsnd_m].nx * object[selsnd_m].dx))
endif

selectObject: selsnd_m

noprogress To Pitch: 0.01, minF0, maxF0

Save as text file: soundDir$ + "/" + calcPitch$
