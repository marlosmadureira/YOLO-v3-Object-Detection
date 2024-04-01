import os
import time
from gtts import gTTS
from playsound import playsound


# Função deletar arquivo mp3 criado
def deleteMp3():
    if os.path.exists('mp3_fp.mp3'):
        os.remove('mp3_fp.mp3')


# Função para converter texto em fala
def fala_resposta(audio):
    speech = gTTS(audio, lang='pt')
    deleteMp3()
    speech.save('mp3_fp.mp3')
    playsound('mp3_fp.mp3')
    time.sleep(0.5)  # Aguarda um breve momento após a reprodução para evitar interrupções
