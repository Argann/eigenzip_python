# -*- coding: utf8 -*-

"""
	EigenZip v0.4
		
		Paramètres :  - chemin vers un dossier composé d'images de mêmes dimensions en .jpg
					  - chemin vers un dossier (vide de préférence)

		Retour : Retourne dans le dossier choisi un fichier .eigmean les eigenvectors, la meanface et les dimensions des images de bases
				 et des fichiers .coef correspondant chacun à une image compressée.

		Usage : python EigenZip.py <pathDuDossierImages> <pathDuDossierDeSortie>

		Exemple : python EigenZip.py faces94/female/ekavaz test/

"""
#Imports pour le traitement de l'image
from PIL import Image
import ImageOps
import numpy

#Imports Divers
import sys
import os
import pickle
import glob
import collections


# ========= FONCTIONS =========

def normaliser(liste, newMin, newMax):
	"""
	Méthode de normalisation de liste.
	Permet de re-borner la liste entre les valeurs newMin et newMax

	:param liste : Liste à normaliser
	:param newMin : borne minimum de la future liste
	:param newMax : borne maximum de la future liste

	:return: La liste redimensionnée entre les bornes demandées
	"""
	liste = numpy.asarray(liste)
	minListe, maxListe = numpy.min(liste), numpy.max(liste)
	#On bascule les valeurs de liste entre [0...1].
	liste = liste - float(minListe)
	liste = liste / float((maxListe - minListe))
	#Et on élargi la plage de donnée entre [newMin...newMax].
	liste = liste * (newMax - newMin)
	liste = liste + newMin
	return liste.T


def calculerCoefficient(eigenVectors, image, meanFace):
	"""
	Méthode permettant de calculer les coefficients du vecteurImage donné par rapport aux eigenVectors et a la meanface donnée.

	:param eigenVectors : vecteurs propres
	:param image : vecteur image dont les coefficients sont à calculer
	:param meanFace : vecteur image contenant la meanface

	:return: Liste contenant les coefficients
	"""
	return numpy.dot(image - meanFace, eigenVectors)


def reconstruireImage(eigenVectors, coefficients, meanFace):
	"""
	Méthode permettant de reconstruire une image à partir de ses coefficients, d'eigenvectors et d'une meanFace

	:param eigenVectors : vecteurs propres
	:param coefficients : coefficients de l'image à reconstruire
	:param meanFace : vecteur image contenant la meanFace

	:return: vecteur image contenant l'image reconstruite
	"""
	return numpy.dot(coefficients, eigenVectors.T) + meanFace


def zipFolder(path_input, path_origin, path_output):
	"""
	Méthode permettant de comprésser avec la méthodes des EigenVectors le dossier d'image dans input, 
	et de mettre en sortie les fichiers .coeff et need.eigmean dans path_output.

	:param path_input : Chemin vers le dossier contenant les images
	:param path_origin : Chemin de lancement du script
	:param path_output : Chemin de sortie du script
	"""

	#Définir ici le nombre d'eigenvector a sauvegarder. Il ne doit pas dépasser le nombre d'images du dossier à compresser.
	#Pour avoir une bonne qualité finale, mettre un minimum de 80 eigenface.
	NUMBER_OF_PRINCIPAL_COMPONANTS = 80 


	# ========= CODE =========
	#Si on arrive là, c'est que tout va bien.
	print " ===Création du visage moyen=== "
	
	os.system("python meanScript.py "+path_input+" temp_meanface.jpg") #CHANGE cCAAAAA

	print " ===DONE=== \n\n"

	print " ===Création de l'Eigenface=== "

	
	os.chdir(path_input) # On change le dossier courant par le dossier d'images

	print "-- Lecture du dossier d'image ... --"
	img_list = []
	nbrImages = 0
	for img in glob.glob(r"*.jpg"): #On parcours tout les fichiers terminant par '.jpg'
		open_image = Image.open(img) #On ouvre l'image
		gray_image = ImageOps.grayscale(open_image) #On la transforme en niveau de gris
		larg, haut = gray_image.size #On capture les dimensions de l'image
		data_image = gray_image.getdata() #On récupère une liste contenant pour chaque 'case' de la liste la valeur en niveau de gris du pixel correspondant de l'image 
		img_list += [data_image] #Finalement, on ajoute les données de l'image dans la liste de données d'images
		nbrImages += 1
	print "DONE\n"
	os.chdir(path_origin) #On revient au chemin d'origine

	print "-- Soustraction de la meanface --"

	im = Image.open("temp_meanface.jpg") #On ouvre la meanface créée précédemment.
	pix = im.load() #Grâce à cette méthode, on a accès directement aux pixels de l'image.
	data_meanface = numpy.array([[pix[x, y] for x in xrange(larg)] for y in xrange(haut)]) #On traduit un objet PixelAccess (utilisé par PIL) vers une liste python compréhensible par numpy
	data_meanface = data_meanface.flatten() #On transforme notre matrice de pixel de l'image meanface en un vecteur

	img_list_without_meanface = numpy.subtract(numpy.array(img_list), data_meanface) # On soustrait la meanface de toutes les images
	
	os.remove("temp_meanface.jpg")
	print "DONE\n"



	print "-- Normalisation des differences faces --"
	#A cause de la soustraction de la meanface, certaines images se retrouvent avec des pixels ayant des valeurs négatives
	#Du coup, on va tout rebasculer entre les bornes [0..255]
	img_list_proper = [] #On crée une liste qui nous servira a stocker nos images normalisées
	for img in img_list_without_meanface: #On parcours la liste de differencefaces...
		img_list_proper += [normaliser(img, 0, 255)] #Et on les normalises les unes après les autres.
	print "DONE\n"

	img_list_proper = numpy.array(img_list_proper) #On transforme la liste d'images "propres" en array compréhensible par numpy

	[imls_x, imls_y] = img_list_proper.shape #On sauvegarde le nombre d'images et le taille de chaque image dans deux variables distinctes


	print "-- Calcul de la matrice de covariance et des Eigenfaces --"
	#Calculer la matrice de covariance est impossible, cela demanderai trop de calculs.
	#On doit passer par un subtil stratagème afin de la retrouver.
	#D'ailleurs, img_list_proper est en fait la transposée de la matrice équivalente dans la formule originale.
	#Ca explique pourquoi le .T n'est jamais au bon endroit
	covariance_matrix = numpy.dot(img_list_proper, img_list_proper.T) #On calcule la "petite" matrice de covariance
	[eig_values, eig_vectors] = numpy.linalg.eigh(covariance_matrix) #On calcule les eigenvector et eigenvalues correspondantes a partir de la matrice de covariance
	eig_vectors = numpy.dot(img_list_proper.T, eig_vectors) #Ce calcul permet de récupérer les "grands" eigenvectors
	for i in xrange(imls_x):
		eig_vectors[:,i] = eig_vectors[:,i]/numpy.linalg.norm(eig_vectors[:,i])
	print "DONE\n"

	print "-- Tri et normalisation des eigenvectors --"
	idx = numpy.argsort(-eig_values) # \
	eig_values = eig_values[idx]	 # |- On trie les eigenvector par valeur décroissante de eigenvalue
	eig_vectors = eig_vectors[:,idx] # /

	eig_values = eig_values[0:NUMBER_OF_PRINCIPAL_COMPONANTS].copy() #On ne garde que les NUMBER_OF_PRINCIPAL_COMPONANTS eig_values
	eig_vectors = eig_vectors[:,0:NUMBER_OF_PRINCIPAL_COMPONANTS].copy() #On ne garde que les NUMBER_OF_PRINCIPAL_COMPONANTS eig_vectors

	print "DONE\n"

	coeff = [calculerCoefficient(eig_vectors, x, data_meanface) for x in img_list] #On crée une liste contenant les coefficiants

	print "-- Sauvegarde dans le fichier --"
	os.chdir(path_output)
	with open("need.eigmean", "wb") as fichier:
		monPickler = pickle.Pickler(fichier)
		donnees = {"vect" : eig_vectors, "mean" : data_meanface, "dim" : (larg, haut)} #crée un dictionnaire contenant les différentes données à enregistrer
		monPickler.dump(donnees) #Et on enregistre ce dictionnaire dans le fichier choisi par l'utilisateur

	cmpt = 0
	for x in coeff:
		print ("Sauvegarde de l'image "+str(cmpt)+" sur "+str(len(coeff)-1))
		path = str(cmpt)+".coeff"
		with open(path, "wb") as fichier:
			monPickler = pickle.Pickler(fichier)
			monPickler.dump(x)
		cmpt += 1
	
	print " === TOUT EST BIEN === "
	#Et c'est fini !


def unZipFolder(path_input, path_origin, path_output):
	"""
	Méthode permettant de décompresser avec la méthode des EigenVectors le dossier input, 
	et de mettre en sortie les images .jpg dans path_output.

	:param path_input : Chemin vers le dossier contenant les .coeff et le need.eigmean
	:param path_origin : Chemin de lancement du script
	:param path_output : Chemin de sortie du script
	"""
	
	os.chdir(path_input) #On se déplace dans le dossier contenant les .coeff et le need.eigmean
	coeff = []#On crée une nouvelle liste contenant tout nos futures coefficients
	for img in glob.glob(r"*.coeff"): #On parcours tout les fichiers terminant par '.jpg'
		with open(img) as fichier:
			monDepickler = pickle.Unpickler(fichier)
			donnee = monDepickler.load() #On charge chaque fichier .coeff les uns après les autres...
			coeff += [donnee] #Et on enregistre ces coefficients dans coeff

	with open("need.eigmean", "rb") as fichier: #On enregistre tout simplement les eigenvectors, la meanface et les dimensions de l'image dans donnee
		monDepickler = pickle.Unpickler(fichier)
		donnee = monDepickler.load()


	#On retire de donnee toutes les infos necessaires
	eigvec = donnee["vect"]
	meanface = donnee["mean"]
	larg, haut = donnee["dim"]

	nbrImages = len(coeff)

	print "--- Nombre d'images à charger : ", nbrImages, " ---\n\n"
	os.chdir(path_origin)
	os.chdir(path_output)
	curseur = 0
	#Pour chaque coefficients, on sauvegarde l'image reconstruite correspondante
	for x in coeff:
		print(curseur, " sur ", nbrImages)
		path = str(curseur)+".jpg"
		print path
		image_finale = Image.new("L",(larg,haut)) #On crée une nouvelle image en niveau de gris ("L") et ayant la bonne taille
		image_finale.putdata(reconstruireImage(eigvec, x, meanface)) #On lui injecte les données de l'image reconstruite
		image_finale.save(fp=path) #Et on sauvegarde l'image finale dans le chemin indiquée a la fin du script.
		curseur += 1

	print "DONE"


def meanFace(path_input, path_origin, path_output):
	"""
	Méthode permettant de calculer l'image moyenne des photos contenues dans le dossier indiqué par path_input.
	L'image finale est enregistrée dans path_output

	:param path_input : Chemin vers le dossier contenant les images dont il faut calculer la moyenne
	:param path_origin : Chemin de lancement du script
	:param path_output : Chemin de la future meanface .jpg 
	"""

	os.chdir(path_input) # On change le dossier courant par le dossier d'images

	print "-- Lecture du dossier d'image ... --"
	img_list = []
	for img in glob.glob(r"*.jpg"): #On parcours tout les fichiers terminant par '.jpg'
		open_image = Image.open(img) #On ouvre l'image
		gray_image = ImageOps.grayscale(open_image) #On la transforme en niveau de gris
		larg, haut = gray_image.size #On capture les dimensions de l'image
		data_image = gray_image.getdata() #On récupère une liste contenant pour chaque 'case' de la liste la valeur en niveau de gris du pixel correspondant de l'image 
		img_list += [data_image] #Finalement, on ajoute les données de l'image dans la liste de données d'images

	print "DONE\n"
	
	print "-- Rendu de l'image finale ... --"
	img_array = numpy.array(img_list) # On transforme la liste de données d'images en array, que l'on peut ainsi manipuler avec numpy (plus puissant !)
	
	mean_data = numpy.mean(img_array, axis=0) #On fait ensuite une moyenne de toutes les données d'images. Cela nous donne ainsi les données de la meanface finale !

	int_mean_data = [int(x) for x in mean_data] #Bon, forcément, une moyenne ça renvoi des float. Nous on veux des int, du coup on cast.

	os.chdir(path_origin) #On revient au chemin d'origine

	image_finale = Image.new("L",(larg,haut)) #On crée une nouvelle image en niveau de gris ("L") et ayant la même taille que nos autres images
	image_finale.putdata(int_mean_data) #On lui injecte les données de la meanface calculées précédemment
	image_finale.save(fp=path_output) #Et on sauvegarde l'image finale dans le chemin indiquée a la fin du script.
	print "DONE\n"


# ========= CODE =========

if __name__ == '__main__':

	OKCMD = ["-help", "-zip", "-unzip", "-mean"]
	HELPMSG ="""
-- EigenZip v1.4 --
Authors : 
Argann BONNEAU
Victoria ROGER
Kévin BRIAND
Romain CUSSONNEAU
-------------------

Usage : python EigenZip.py [options] <[path_input] [path_output]>

python EigenZip.py -help
	Affiche l'aide de EigenZip

python EigenZip.py -zip [path_input] [path_output]
	Compresse les images du dossier de path_input dans le dossier path_output

python EigenZip.py -unzip [path_input] [path_output]
	Décompresse les fichiers contenus dans path_input dans le dossier path_output

python EigenZip.py -mean [path_input] [path_output.jpg]
	Calcule la meanface des images contenues dans path_input et l'enregistre dans path_output.jpg

"""
	if len(sys.argv) <= 1:
		print "Pas assez d'arguments donnés, veuillez consulter l'aide avec python EigenZip -help"
	elif sys.argv[1] not in OKCMD:
		print "Argument non valable, veuillez consulter l'aide avec python EigenZip -help"
	elif sys.argv[1] == "-help":
		print HELPMSG
	elif sys.argv[1] == "-mean":
		if len(sys.argv) != 4:
			print "Pas assez d'arguments donnés, veuillez consulter l'aide avec python EigenZip -help"
		elif not os.path.isdir(sys.argv[2]):
			print "Le premier argument donné devrait être un dossier. Veuillez consulter l'aide avec python EigenZip -help"
		elif sys.argv[3][-4:] != ".jpg":
			print "Le deuxième argument devrait se finir par .jpg. Veuillez consulter l'aide avec python EigenZip -help"
		else:
			meanFace(sys.argv[2], os.getcwd(), sys.argv[3])
	elif sys.argv[1] == "-zip":
		if len(sys.argv) != 4:
			print "Pas assez d'arguments donnés, veuillez consulter l'aide avec python EigenZip -help"
		elif not os.path.isdir(sys.argv[2]):
			print "Le premier argument donné devrait être un dossier. Veuillez consulter l'aide avec python EigenZip -help"
		elif not os.path.isdir(sys.argv[3]):
			print "Le second argument donné devrait être un dossier. Veuillez consulter l'aide avec python EigenZip -help"
		else:
			zipFolder(sys.argv[2], os.getcwd(), sys.argv[3])
	elif sys.argv[1] == "-unzip":
		if len(sys.argv) != 4:
			print "Pas assez d'arguments donnés, veuillez consulter l'aide avec python EigenZip -help"
		elif not os.path.isdir(sys.argv[2]):
			print "Le premier argument donné devrait être un dossier. Veuillez consulter l'aide avec python EigenZip -help"
		elif not os.path.isdir(sys.argv[3]):
			print "Le second argument donné devrait être un dossier. Veuillez consulter l'aide avec python EigenZip -help"
		else:
			unZipFolder(sys.argv[2], os.getcwd(), sys.argv[3])


