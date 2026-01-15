#Python'daki standart listelere (list) benzerler ancak matematiksel
# işlemler ve büyük veri setleri için çok daha güçlü ve hızlıdırlar.
import numpy as np

import pandas as pd

df = pd.read_csv(r"C:\Users\asus\PycharmProjects\SpotifyProject2\dataset - dataset.csv.csv")


#print(df.head())

#1.adım olarak descrpitive statistics incelemeri yapıyorum

#sayısal değişkenler için ayrı bir tanımlama yapıyorum

#eski halinde stringleri aldığı için bu kodu tercih ettim
num_var = df.select_dtypes(include=[np.number]).columns.tolist()
#print(num_var)

#temel descriptive fonksiyonların list'liyorum

desc_agg=['sum', 'min', 'std', 'var', 'mean' ,'max']

#bu fonksiyonları sayısal değerlere uyguluyorum

desc_agg_dict={col:desc_agg for col in num_var}
#print(desc_agg_dict)

desc_summ=df[num_var].agg(desc_agg_dict)

#desc_summ'ı print ediyorum, böylece her değişkenin toplam, ortalama, standart sapma, min ve max değerlerini inceleyebilirim
#print(desc_summ)

#numpy array'e dönüştürmek istiyorum

df_desc_na=np.array(desc_summ)
#print(df_desc_na)

#df numpy array olarak kullanılmak isterse;vektörel işlemler vb.

df_na=np.array(df)
#nt(df_na)
#Overview devam

import seaborn as sns
#print(df.shape)

#print(df.info())

#print(df.columns)
#SUNUMDA NELER ÇIKTIĞINI YAZMAYI UNUTMA




#missing value için ekstra kontrol yapıyorum FALSE geliyor
#print(df.isnull().values.any())

#her bir değişkene ait descriptive analytics değerleri bir tabloya yeniden yazdırıyorum
desc_summy2 = df.describe().T
#print(desc_summy2)
#Tüm değişkenlerde count değeri 113.999.
#duration_ms yüz binlerle ifade edilirken, danceability 0 ile 1 arasında, loudness ise -60 ile +4 arasında değişiyor.
                                                 #|
                                                #\/
#Eğer bu veriyi K-Means, KNN veya Sinir Ağları gibi "mesafe tabanlı" makine öğrenmesi modellerine sokacaksan,
#kesinlikle ölçeklendirme (Normalization/Standardization) yapmalısın. Aksi takdirde duration_ms baskın gelir, diğer özellikler yok sayılır.

target = 'popularity'

#Target'ı inceleyelim

#print(df[df[target] > df[target].mean()][target].count())
#print(df[df[target] < df[target].mean()][target].count())
#makine öğrenmesi modelleri için harika
#Linear Regression iyi olur!!!
#data oldukça dengeli ama dağılımı U şeklinde olabilir bu yüzden seaborn kullanmalıyız


#print(df.loc[df[target] > df[target].mean(), 'danceability'].head())

#bütün sensör verilerinde Target değişkeninde paralel pozitif çarpık sensörleri inceliyorum


sensor = df.iloc[: , 8:20]
#print(sensor)

#print(sensor.columns)

#değişkenlerin görseller ile incelenmesi

from matplotlib import pyplot as plt

#değişkenlerin grafiklerini çıkarıyorum

sns.boxplot(x=sensor['danceability'])
#plt.show()

def num_summary(sensor, numerical_col, plot=True):
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

#    print(sensor[numerical_col].describe(quantiles).T)
#    print("\n")

    if plot:
        sensor[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        #plt.show(block=True)

#print(num_summary(sensor, 'danceability', plot=True))




#tüm değikenler için bir kod ile grafikler üretiyorum

#for col in sensor:
  #num_summary(sensor, col, plot=True)

#tüm veri seti sayısal olduğu için, değiken yakalamak veya genelleme sürecini atlıyorum

#bağımlı değişkenin bağımsız değişkenler üzerinden analiz ediyorum

#print(df.groupby('popularity')['danceability'].mean())
#"Şarkı ne kadar dans edilebilirse, popüler olma şansı o kadar artıyor" hipotezini doğruluyor. Dans edilebilirlik ile popülerlik arasında pozitif bir ilişki var.

#tüm sayısal değerler için bu fonskiyonu itere etmek istiyorum

#def target_summary_with_num(dataframe, target, num_col):
# print(dataframe.groupby('popularity').agg({num_col: 'mean'}), end='\n\n\n')

#target_summary_with_num(df, 'popularity', 'danceability')   #"Şarkının dans edilebilirlik puanı 0.50'lerden 0.70'lere çıktığında, popüler olma şansı artıyor."
#print(dataframe.groupby('popularity').agg({num_col: 'mean'}), end='\n\n\n') ile aynnı sonucu verir


  #fonksiyon tüm columnlar için de çalışsn
#for col in sensor:
    #target_summary_with_num(df, target ,col)

    #peki sensorler arasında korelasyon nasıl, model optimizasyonu için korelasyona göre ayıklamayı değerlendiriyorum

    #tüm korelasyonları çıkayorum
#corr = sensor.corr()

#korelasyon ısı haritası çıkarmak istiyorum
#plt.figure(figsize=(12, 12))
#sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f", vmin=-1, vmax=1)
#plt.show()  #"Spotify verisetimde Enerji ve Gürültü %76 oranında koreledir. Akustik şarkılar ise %73 oranında düşük enerjiye sahiptir.
# Bu yüzden akustik şarkıların popüler (hit) olma şansı, günümüzün 'gürültülü' trendlerine ters düştüğü için daha düşüktür."
#Dans edilebilirlik artınca popülerlik artıyor.
#Enstrümantal şarkıların popülerliği neredeyse sıfır.
#Gürültülü (Loudness) şarkılar daha popüler.



#sensorlerin 3'lü gruplar halinde korelasyonlarının ortak olduğunu fark ediyorum
#ayrıca korelasyonun yüksek olduğu değişkenlerin de ayıklamak istiyorum, tekilleştirmek burada overfitting'i engeller

#cor_matrix = sensor.corr().abs()
#df.corr(): İlişkileri bulur (-1 ile +1 arası).
#.abs() (Mutlak Değer): Eksileri artı yapar.
#Neden Gerekli? Çünkü "Multicollinearity" (Çoklu Bağlantı) sorunu için -0.90 (Güçlü Negatif İlişki) ile +0.90 (Güçlü Pozitif İlişki) aynı derecede tehlikelidir.
#İkisini de yakalamak istiyoruz.
#ANCAK BUNU DF ÜZERİNDE YAPARSAK İLİŞKİLİ OLANI DA SİLEBİLİR BU YÜZDEN SENSOR ÜZERİNDE YAPILMALI

#köşegenlerin altındaki tüm simetrik değerleri siliyorum, korelasyon matrisini tekil ve mutlak değerli yapıya dönüştürüyorum
#.90 üzerini listemden çıkaracağım

#upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
#Matrisin sadece üst yarısını al (Çünkü matris simetriktir, aynısını iki kere kontrol etmeyelim

#drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
#  Belirlediğin sınırdan (örneğin %90) yüksek olan sütunları bul
# Eğer iki değişken %90 üzeri benzerse, birini silmek yeterlidir.

#cor_matrix[drop_list]

#veri setlerimden (hem) yoğun korele değişkenleri siliyorum
#df = df.drop(drop_list, axis=1)
#sensor = sensor.drop(drop_list, axis=1)
#(İstersen) Bu değişkenleri ana veriden düşür

#print(sensor.shape)  #"Multicollinearity analizi için %90 eşik değeri belirledim. Ancak veri setimde en yüksek korelasyon 0.76 (Energy-Loudness) çıktığı için, hiçbir değişken kritik sınırın üzerinde çakışmadı.
#                       Bu yüzden bilgi kaybı yaşamamak adına 12 değişkeni de modelde tuttum."
#en yüksek benzerlik 0.76 olduğu için, belirlediğin 0.90 sınırına (threshold) kimse takılmadı.
#Yani kodun dedi ki: "Bu değişkenlerin hepsi birbirinden yeterince farklı, hepsini tutuyorum."




#MODEL DEVELOPMENT

#Modeling
#Prediction
#Evaluation
#Hyperparameter Optimization
#Finalization

#Roadmap olarak bu şekilde ilerliyor olacağım

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
#pd.set_option('display.max_columns', None) #veriyi küçülttüğüm için kaldırıyorum




#Y = df.iloc[:, 1:2]
X = sensor # Daha önce ayırdığımız 12 sensör verisi


#bağımsız değişkenleri standardize ediyoruz
#STANDARDİZASYON (K-Means için ZORUNLUDUR!)
# KMeans mesafeye bakar. Loudness ile Tempo arasındaki farkı eşitlemek şart.

X_scaled = StandardScaler().fit_transform(X)
# Pandas DataFrame haline getirelim (Yellowbrick kütüphanesi için daha sağlıklı)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

#öznitelikleri(feature kümeleme yapmak istiyorum, --clusterin modelling
#sonra her bir satırın yanına sınıfını ekleyeyip, regresyon modeli oluşturacağım)
#PCA da denenebilir
import matplotlib.pyplot as plt
import yellowbrick as yb
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#optimum küme sayısını belirliyorum

kmeans = KMeans()  #kmeans modelini başltıyoruz
elbow = KElbowVisualizer(kmeans, k=(2,20))
 #görselleştiriciyi kuruyoruz 2 ile 20 küme arasında
elbow.fit(X_scaled_df)
#Modeli görselleştiriciye oturtuyoruz
#elbow.show()
#"114.000 şarkılık veri setini matematiksel olarak EN İYİ gruplama sayısı 10'dur."
#insanların eliyle etiketlediği türler (Pop, Jazz, Rock) dışında, ses özelliklerine (Danceability, Energy vs.) dayalı 10 farklı "gizli profil" var.
#print(elbow.elbow_value_)
# 11 çıktı tabloda ise 10 çıktı ama 11 kullanılacak


#bu cluster sayısı ile k-means çalıştırıyoruz

kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state = 17).fit(X_scaled_df)
kmeans.get_params()

#print(kmeans.n_clusters)
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#print(kmeans.inertia_)

#label'ları cluster olarak tanımladım Hesaplanan sınıfları (0'dan 11-12'e kadar olan etiketleri) alıyoruz
clusters = kmeans.labels_

#0'dan başlamamalarını istiyorum- cluster id'ler 1-12 arası olacak
sensor['cluster'] = clusters + 1

#print(sensor.head)  # tam olarak istediğim sonucu göstermediği için alttakini kullanacağım

#print(sensor.iloc[:, -3:].head()) #cluster'ın sonuna eklendiğini görmek için yaptım



#print(sensor.columns)

#Index(['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
#       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
#       'time_signature', 'cluster'],
#      dtype='object')



#BURADAN SONRASI 11 AYRI CLUSTER MODELİ TRAIN EDİP YENİ DEĞERLERDE EN İY ÇALIŞANI MODEL OLARAK kaBUL ETMEYİ TERCİH EDİYORUM

#print(sensor.groupby('cluster').agg(['count', 'mean', 'median']))  #"Verinin Nüfus Sayımı" Cluster 6: En kalabalık grup (21.669 şarkı).Cluster 5: En küçük grup (1.075 şarkı).


#train veri setinin kümelerini bir csv dosyasına yazalım
sensor.to_csv('cluster.csv')

#cluster'lar incelediğimde temel bileşenler analizi yapmak istiyorum PCA
#pca = PCA()
#pca_fit = pca.fit_transform(X_scaled_df)
#print(pca_fit )   # pca benim verisetim için pek mantıklı bir karar değil clustering yapmayı hedefliyorum


#açklanan varyans oranı
#pca.explained_variance_ratio_
#Veri setinin yarısını bile açıklamak için 3 farklı bileşene ihtiyacın var. Bilgi tek bir yerde toplanmamış, tüm özelliklere (Danceability, Energy, Tempo...) dağınık halde yayılmış.
#bilgi bu kadar dağınıkken, PCA yapıp veriyi şifrelemek hem bilgi kaybettirir hem de yorumlama gücünü elinden alır.

#np.cumsum(pca.explained_variance_ratio_)
#ilk 9 değişenle %92'sini kurtarıyorum

#pca = PCA().fit(X_scaled_df)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Bileşen Sayısı')
#plt.ylabel('Kümülatif Varyans Oranı')
#plt.show()

#PCA Modeli

#pca = PCA(n_components=9)
#pca_fit = pca.fit_transform(X_scaled_df)
#pca.explained_variance_ratio_

#print(np.cumsum(pca.explained_variance_ratio_))

#####################İLETİLEN TEST VERİSİNE UYGULUYORUM

df_test_PCA = pd.read_csv(r'C:\Users\asus\PycharmProjects\SpotifyProject2\dataset - dataset.csv.csv')

X_test_targetsiz_PCA = df_test_PCA.drop(['track_name', 'popularity', 'cluster', 'segment', 'track_id', 'artists', 'album_name', 'explicit', 'Unnamed: 0', 'track_genre', 'genre'], axis=1, errors='ignore')
#Hata veren 'acoustic' kelimesinin bulunduğu sütunu da (track_genre) çöpe atıyor.
#errors='ignore' sayesinde, eğer sütunun adı genre ise onu siler, track_genre ise onu siler; hangisi yoksa onu dert etmez.

#X_test_targetsiz_PCA = X_test_targetsiz_PCA.dropna()

#pca_fit = pca.fit_transform(X_test_targetsiz_PCA)
#print(pca.explained_variance_ratio_)  #verinin bilgi yükü tüm sütunlara dağılmış durumda.her bir bileşenin tek başına ne kadar bilgi taşıdığını verir.
#veri setimizdeki varyans (bilgi) tek bir boyutta toplanmamış, dengeli dağılmış. %90 bilgi koruma oranına ulaşmak için en az 9 bileşene ihtiyacımız var.
# Bu yüzden boyut indirgeme yapmak (PCA) bu veri setinde büyük bir avantaj sağlamıyor.


#Cumulative Sum Yani listedeki sayıları üst üste toplayarak ilerler."Nerede duralım?" kararını vermemizi sağlar.
# Listeye bakarız ve sayı 0.90'ı (yani %90'ı) geçtiği anda "Tamam, bu kadar bileşen yeterli" deriz.Bu kod için 9 bileşen
#np.cumsum(pca.explained_variance_ratio_)

#print(X_test_targetsiz_PCA.head()) # veri tamamen sayısal hale geldi


#RMSE_PCA = np.cumsum(pca.explained_variance_ratio_)
#print(RMSE_PCA)
#[0.99999992 1.         1.         1.         1.         1.
# 1.         1.         1.        ]     #0.99 sonucu, veriler arasındaki boyut farkından (ms vs 0-1) kaynaklanan bir hataydı. scale etmemiz gerekiyor


#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_test_targetsiz_PCA)

#pca = PCA(n_components=9)
#pca_fit = pca.fit_transform(X_scaled)

#print(np.cumsum(pca.explained_variance_ratio_))
#[0.22808168 0.34610493 0.44368129 0.53309187 0.60867482 0.67845211
 #0.74654495 0.8109195  0.87176938]  -----> 12 taneden 9 tanesi matematiksel olarak yetersiz değil ama verimsiz. Orijinal verilerle devam etmeliyiz PCA gereksiz


#pca_df = pd.DataFrame(pca_fit, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9'])
#print(pca_df)

#"Veri setindeki boyutları indirgemek amacıyla PCA uygulanmış ve 12 değişken, %87 varyansı açıklayan 9 temel bileşene (PCA1-PCA9) dönüştürülmüştür."
#final_df_targetsiz = pd.DataFrame(pca_fit, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9'])
#print(final_df_targetsiz) # sonuçta - + olmasının negatif bir anlamı yok çünkü Bunlar koordinatlardır.
# Bu sayılar StandardScaler (ölçekleme) işleminden geçtiği için 0'ın etrafında (pozitif ve negatif) dağılmıştır

#bu PCA sonuçlarının yanına şarkı isimlerini (track_name) veya eski ID'lerini geri yapıştırmaksa, o zaman concat kullanılır.
#final_df_with_names = pd.concat([df_test_PCA[['track_name']].reset_index(drop=True), final_df_targetsiz], axis=1)
#print(final_df_with_names)



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

X_test_targetsiz_PCA = X_test_targetsiz_PCA.dropna()  #Veri setimdeki eksik (boş/NaN) satırları sil

 #veriyi eğitiyoruz

y = df_test_PCA.loc[X_test_targetsiz_PCA.index, 'popularity']  # Popülerliği ana dosyadan çekiyoruz ve x ve y eşit olmalııı
reg_model = LinearRegression()  #Modeli Tanımla ve Eğit
reg_model.fit(X_test_targetsiz_PCA, y)  # Modelimi elimdeki verilerle eğitiyorsun

y_pred_test_targetsiz_PCA = reg_model.predict(X_test_targetsiz_PCA)
#print(y_pred_test_targetsiz_PCA[:5])  # eğitilen verinin ilk 5 şarkısı test ediliyor
#[33.81874509 34.43549908 37.21262892 35.41347146 38.35797769]

from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y, y_pred_test_targetsiz_PCA))  #hata payı hesaplanıyor
r2 = r2_score(y, y_pred_test_targetsiz_PCA)  #odel, popülerlik değişimini yüzde kaç açıklayabiliyor?
#print(f"Ortalama Hata Payı (RMSE): {rmse}")  #Ortalama Hata Payı (RMSE): 22.044327427619322
#print(f"Model Başarısı (R2 Score): {r2}")   #Model Başarısı (R2 Score): 0.023224733093134775
#Modelin yaptığı tahminler, gerçek değerden ortalama 22 puan sapıyor.
#bir şarkının popüler olup olmayacağını açıklama konusunda sadece %2.3 başarılı.
#LİNEAR REGRESSION UYGUN DEĞİL




#PCA olmadan model deniyorum


df = pd.read_csv(r'C:\Users\asus\PycharmProjects\SpotifyProject2\dataset - dataset.csv.csv')

# 2. X ve y OLARAK AYIR (Temizlik burada yapılır)
# Unnamed: 0 ve diğer sözel sütunları atıyoruz
drop_listesi = ['popularity', 'track_name', 'track_id', 'artists',
                'album_name', 'explicit', 'track_genre', 'genre', 'Unnamed: 0']

y = df['popularity']
X = df.drop(drop_listesi, axis=1, errors='ignore').select_dtypes(include=[np.number])

# 3. VERİYİ BÖL (Train ve Test diye burada ayırıyoruz)
# Bu işlem rastgele %20'sini test için kenara ayırır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# DECISION TREE UYGULA
dt_model = DecisionTreeRegressor(random_state=42)

# Modeli Eğit
dt_model.fit(X_train, y_train)

# TAHMİN VE SONUÇ
y_pred_tree = dt_model.predict(X_test)

rmse_sonuc = round(np.sqrt(mean_squared_error(y_test, y_pred_tree)), 2)
r2_sonuc = round(r2_score(y_test, y_pred_tree), 3)

#print("Decision Tree RMSE (Hata Payı):", rmse_sonuc)   #Modelin bir şarkının popülerliğini tahmin ederken ortalama 20 puan yanılıyor.
#print("Decision Tree R2 (Başarı Oranı):", r2_sonuc)  #Modelin, şarkıların popülerliğindeki değişimin %14.5'ini açıklayabiliyor.

#Tek bir Karar Ağacı (Decision Tree), Spotify popülerliğini çözmek için yetersiz kalıyor.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rmse_rf = round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 2)
#print("Random Forest RMSE (Hata Payı):", rmse_rf) #Hata payı decision tree ye göre 2 puandan fazla düştü. Model artık daha isabetli tahminler yapıyor.
r2_rf = round(r2_score(y_test, y_pred_rf), 3)
#print("Random Forest R2 (Başarı Oranı):", r2_rf)  #bir şarkının neden popüler olduğunu sadece %33 oranında açıklayabiliyor.

############## tahmin kısmı
# Modelin içinden önem puanlarını çekiyoruz
onem_df = pd.DataFrame({
    'Ozellik': X_train.columns,
    'Etki_Puani': rf_model.feature_importances_
})

#print("En önemli 5 özellik:")
#print(onem_df.sort_values(by='Etki_Puani', ascending=False).head(5))

#Akustik ayarı, Dans edilebilirliği ve Süresi en kritik üçlüdür (Liveness, Instrumentalness vb.) daha az önemlidir.




















