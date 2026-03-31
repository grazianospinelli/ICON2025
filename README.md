# Icon24-25
Repository per il progetto di Ingegneria della Conoscenza realizzato da:
- Spinelli Graziano (mat.322084)

# Esecuzione 
## Fase iniziale
Installare SWIProlog (installare la versione a 64 bit)

<code>https://www.swi-prolog.org/download/stable/bin/swipl-10.0.2-1.x64.exe.envelope</code>

Creare una cartella, scaricare il codice e creare l'ambiente virtuale

<code> cd Icon </code>

<code> python -m venv Icon </code>

Installare le dipendenze:

<code>pip install -r requirements.txt</code>

## PARTE 1 - Knowledge Base PROLOG

Creazione Knowledge Base: <br>

<code> python Make_Pairing_KB.py </code>
 
 Importare il file ottenuto nella cartella datasets kb.pl in SWI-Prolog <br>
 Testare la KB nell'ambiente di SWI-Prolog ponendo le query desiderate. <br>

## PARTE 2 - Apprendimento Supervisionato

<code> python Wine_Rating.py</code>
 
 


