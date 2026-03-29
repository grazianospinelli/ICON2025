import warnings
warnings.filterwarnings("ignore")
import time
import kagglehub
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Scarico DATASET da KAGGLE
path = kagglehub.dataset_download("budnyak/wine-rating-and-price")
print("Path to dataset files:", path)

red = pd.read_csv(path+'\\Red.csv')
white = pd.read_csv(path+'\\White.csv')
sparkling = pd.read_csv(path+'\\Sparkling.csv')
rose = pd.read_csv(path+'\\Rose.csv')

# Introduzione di una colonna 'WineStyle'
# con tipologia vino per ogni dataset
red['WineStyle'] = 'red'
white['WineStyle'] = 'white'
sparkling['WineStyle'] = 'sparkling'
rose['WineStyle'] = 'rose'

# Fusione dei 4 dataset in un unico dataset
wines =  pd.concat([red, white, sparkling, rose], ignore_index=True)

print("##############################################################")
print("1 - Visualizzazione preliminare Dataset")
print(wines.head())
print(wines.columns)
print(wines.shape)
print(wines.info())

# Il dataset contiene vini dove l'annata non è specificata e ha valore N.V., 
# di solito l'annata incide sul prezzo e/o sul rating del vino 
# eliminiamo quei vini dal dataset, riscriviamo indice e trasformiamo il campo Year in intero
wines = wines[wines['Year'] != "N.V."]
wines = wines.reset_index(drop=True)
wines['Year'] = wines['Year'].astype('int')

# Eliminiamo le colonne Name, Region e Winery perchè difficili da categorizzare 
# anche se solitamente l'importanza di una cantina o la regione di provenienza di un vino influiscono sul prezzo o sul rating
wines = wines.drop(['Name', 'Region', 'Winery'], axis=1)

# Dei due campi categorici del dataset restanti WineStyle ha 4 valori e si presta bene ad un one-hot encoding
# Il numero di nazioni diverse contenute nel dataset è 33, davvero al limite per effettuare un one-hot encoding
# ma si è deciso comunque di non perdere questa feature che come la regione di provenienza influisce sul prezzo del vino
# get_dummies è l'operatore pandas per one-hot encoding
# solitamente questa operazione andrebbe fatta dopo la suddivisione in training set e test set 
# per evitare contaminazioni del test set (data leakage)
print("Cardinalità Country: " + str(wines.Country.nunique()))
wines = pd.get_dummies(wines, columns = ['WineStyle'])
wines = pd.get_dummies(wines, columns = ['Country'])

# Visualizzazione e rimozione OUTLIERS
# Vini che hanno un punteggio molto basso ma con un prezzo nella media
print(wines[(wines['Rating'] < 2.8) & (wines['Price'] > 7)])
wines.drop(wines[(wines['Rating'] < 2.8) & (wines['Price'] > 7)].index, inplace=True)
# Unico Vino con un prezzo molto più alto rispetto a tutti gli altri della sua categoria
print(wines[(wines['Price'] > 2900)])
wines.drop(wines[(wines['Price'] > 2900)].index, inplace=True)

print("##############################################################")
print("2 - Visualizzazione Dataset dopo PREPROCESSING")
print(wines.head())
print(wines.columns)
print(wines.shape)
print(wines.info())
print(wines.describe())

# DISEGNA HEATMAP DELLE CARATTERISTICHE
# si evince che la maggior correlazione c'è tra Rating e Price
corrs = wines[['Rating','NumberOfRatings','Price','Year']].corr() 
fig, ax = plt.subplots(figsize=(7,5))        
sns.heatmap(corrs,annot = True,ax=ax,linewidths=.6, cmap = 'YlGnBu')
plt.savefig('1-wines_correlation.png')
plt.close()

# DISEGNA GRAFICO RAPPORTO PREZZO, RATING
plt.figure(figsize=(13,5))
graph = sns.regplot(x=np.log(wines['Price']), y='Rating', data=wines, fit_reg=False, color='blue')
graph.set_title("Distribuzione Rating x Price", fontsize=20)
graph.set_xlabel("Price(EUR)", fontsize= 15)
graph.set_ylabel("Rating", fontsize= 15)
graph.set_xticklabels(np.exp(graph.get_xticks()).astype(int))
plt.savefig('2-Rating-Price.png')
plt.close()
# plt.show()

################################################################
# Selezioniamo la feature target Y: Rating
y = wines.Rating

# Selezioniamo le Feature X
cols = [col for col in wines.columns if col != 'Rating']
X = wines[cols]

# Suddividiamo il dataset in traing set e test set tenendo il 30% degli esempi per il test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

#################################################################################
# Funzione per testare modello e disegnare grafico scatter
def plot_model(name,model, X_train, y_train, y_test):
    
    # Facciamo il fit del modello sul training set (Addestramento)
    model.fit(X_train, y_train)

    # Facciamo predizione sul test set
    y_pred = model.predict(X_test)

    # Mostriamo Metrica Loss L1 Media per Modello
    mae = mean_absolute_error(y_test, y_pred)
    # print(f"{name} MAE: {mae:.5f}")

    # Disegnamo Scatter plot: y_test vs y_pred    
    plt.scatter(y_test, y_pred, alpha=0.7, color='red', label=f'MAE: {mae:.4f}', s=30)    
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', lw=2)  # Linea ideale
    plt.xlabel('Valori Reali (y_test)')
    plt.ylabel('Predizioni (y_pred)')
    # plt.text(5, 5, f'MAE: {mae:.4f}')
    plt.title(name+' - Predizioni vs Valori Reali')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(name+'-Scatter-plot.png')
    plt.close()
    # plt.show()    

#################################################################################
# Funzione per disegnare curve apprendimento
def plot_learning_curves(name, model, X, y):
    # Aggiungi train_sizes espliciti per avere più punti sulla curva
    # np.linspace(0.1, 1.0, 10) usa dal 10% al 100% del dataset
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=10, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        n_jobs=-1 # Velocizza il calcolo usando tutti i core della CPU
    ) 

    # Calcola medie 
    mean_train_errors = -np.mean(train_scores, axis=1)
    mean_test_errors = -np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(12, 8)) 
    plt.plot(train_sizes, mean_train_errors, 'o-', label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, 'o-', label='Errore di testing', color='red')
    
    # Aggiungiamo una griglia per leggere meglio i valori
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.title(f'Curva di apprendimento per {name}')
    plt.xlabel('Dimensione del training set (n. campioni)')
    plt.ylabel('Mean Squared Error (L2)')
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.savefig(f'{name}-Learning-curve.png')    
    plt.close() # Pulisce la memoria

###################################################################################################
# MODEL SELECTION

# Lista modelli da adattare
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(criterion='absolute_error', random_state=1),    
    'RandomForestRegressor': RandomForestRegressor(n_estimators=150, random_state=1),    
    'GradientBoosting': GradientBoostingRegressor(n_estimators=150, max_depth=6, min_samples_split=5, learning_rate=0.02, loss="squared_error", random_state=1)
}

results = []
for name, model in models.items():
    print(f'Lavorando su modello {name} ...')

    # cross_val_score esegue K-Fold Cross Validation automaticamente su 20 fold
    # e restituisce un array con lo score di ogni fold.
    # usiamo metrica R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - y_mean)²)
    cv_scores = cross_val_score(model, X_train, y_train, cv=20, scoring='r2')
    # Facciamo la media sulle 20 iterate
    mean_cv = cv_scores.mean()
        
    # Plot learning curve per modello
    plot_learning_curves(name, model, X_train, y_train)
    
    # Plot grafico scatter modello
    plot_model(name, model, X_train, y_train, y_test)
    
    results.append({
        'Modello': name,
        'CV Mean Accuracy': f"{mean_cv:.5f}",
        'CV Std': f"{cv_scores.std():.5f}",        
    })


# Disegna Grafico di Comparazione ad Istogrammi
models_names = [r['Modello'] for r in results]
mean_cvs = [float(r['CV Mean Accuracy']) for r in results]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(models_names, mean_cvs, color='#85B7EB', zorder=3)
ax.set_title('Confronto CV Mean R² tra modelli', fontsize=13, pad=12)
ax.set_xlabel('Modello')
ax.set_ylabel('CV Mean R²')
ax.set_ylim(min(mean_cvs) - 0.05, max(mean_cvs) + 0.05)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('Model_Comparison.png')
plt.close() 

# Print Risultati
df = pd.DataFrame(results).sort_values('CV Mean Accuracy', ascending=True)
print(df)

# Trova l'indice della riga con Accuracy massima
idx_max = df['CV Mean Accuracy'].idxmax()

# Ottieni il nome del Modello corrispondente con la prestazione massima
modello_max = df.loc[idx_max, 'Modello']
print("Il Modello con Accuratezza media massima è: "+modello_max)

time.sleep(5)
################################################################################
# IPERPARAMETERS SELECTION

# Gradient Boosting è il modello che ha avuto le migliori performance su questo dataset 
# con Deviazione Standard < 0.02 e R² medio più alto di tutti.
# Data una griglia di iperparametri andremo a fare Cross Validation 
# per selezionare gli iperparametri migliori da assegnare a Gradient Boosting Regressor
# Grid dei parametri del modello da testare
param_grid = {
    'n_estimators': [150, 200, 250, 300],
    'learning_rate': [0.05, 0.08, 0.1, 0.15],
    'max_depth': [2, 3, 4 ],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2]
}

# Modello base
gb_regressor = GradientBoostingRegressor(loss="squared_error", random_state=1)

# GridSearch con 10-fold cross-validation
grid_search = GridSearchCV(
    estimator=gb_regressor,
    param_grid=param_grid,
    cv=10,
    scoring='r2',
    n_jobs=-1,  # Usa tutti i core CPU
    verbose=0
)

# Addestra GridSearch su Gradient Boost
grid_search.fit(X_train, y_train)

# Migliori parametri trovati
print("Migliori parametri:", grid_search.best_params_)
print("Miglior punteggio CV:", grid_search.best_score_)

params=grid_search.best_params_

################################################################################
# RANDOM RESTART CYCLE del modello Gradient Boost Regressor 
# con i migliori parametri selezionati, partendo da punti diversi 
# del training set modificando il parametro seed e usando il parametro
# subsample=0.8 che se minore di 1 attiva Stochastic Gradient Boosting

# Liste per risultati
r2_scores = []
# Dizionario con tutte le previsioni avente per ogni previsione indice pari a metrica R2
all_predictions = {}  
seeds = range(30)

print("Eseguo 30 fit/predict con diversi random_state...")

# Ciclo 30 fit
for seed in seeds:
    # Nuovo modello per ogni seed
    bestgb_model = GradientBoostingRegressor(
        random_state=seed,
        subsample=0.8,
        n_estimators=params['n_estimators'], 
        max_depth=params['max_depth'], 
        min_samples_split=params['min_samples_split'], 
        learning_rate=params['learning_rate'], 
        loss="squared_error"
    )
    
    # Fit e predict    
    bestgb_model.fit(X_train, y_train)
    y_pred = bestgb_model.predict(X_test)
    
    # Metriche
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
    all_predictions[r2]=y_pred
    print(f"Seed {seed}: R² Test = {r2:.4f} - MAE: {mae:.4f}")

# Individuiamo la predizione con metrica R2 massima. 
# La R2 migliore è pari a 1 e si ha quando valori predetti = valori test set
r2_max = max(all_predictions)  # Trova la chiave R² massima
y_best = all_predictions[r2_max]

# Risultati finali
print(f"\nRISULTATI:")
print(f"R² medio singoli:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"R² migliore:        {r2_max:.4f}")

comparison_df = pd.DataFrame({
    'Attuali': y_test,
    'Predetti': y_best
})
print(comparison_df.head(10))

# Grafico 1: Evoluzione R² per seed
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(seeds, r2_scores, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=np.mean(r2_scores), color='red', linestyle='--', 
           label=f'Media R² = {np.mean(r2_scores):.3f}')
plt.axhline(y=r2_max, color='green', linestyle='--', 
           label=f'Miglior R² = {r2_max:.3f}')
plt.xlabel('Seed (Restart #)')
plt.ylabel('R² Test/Pred')
plt.title(f"Evoluzione R² per {len(seeds)} Restart")
plt.legend()
plt.grid(True, alpha=0.3)

# Grafico 2: Previsioni vs Attuali
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_best, alpha=0.3, color='green', label=f'Miglior R²={r2_max:.3f}', s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Valori Reali')
plt.ylabel('Previsioni')
plt.title(f"Miglior Previsione")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Final-plot.png')
plt.close()
# plt.show()

# Grafico di Devianza per GradientBoost
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(bestgb_model.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Curva Apprendimento Gradient Boosting")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    bestgb_model.train_score_,
    "b-",
    label="Errore Training Set",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Errore Test Set"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("LOSS L2 - Mean Squared Error")
fig.tight_layout()
plt.savefig('Learning-Curve-plot.png')
plt.close()

exit()