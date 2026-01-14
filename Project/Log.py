import sys

class Logger(object):
    
    def __init__(self, cartella_attuale, percorso_log):
        '''L'output standard viene
        salvato come self.terminal;
        modalità "w" (overwrite),
        modalità "a" (append)
        '''
        self.percorso_log = percorso_log
        self.cartella_attuale = cartella_attuale
        self.terminal = sys.stdout
        self.log = open(self.percorso_log, "a", encoding="utf-8")

    def write(self, message):
        '''terminal.write scrive nella shell di Pyzo;
        log.write scrive nel file di testo;
        log.flush forza la scrittura immediata su disco
        '''
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        '''Si assicura che non
        ci siano dati non scritti
        '''
        self.terminal.flush()
        self.log.flush()