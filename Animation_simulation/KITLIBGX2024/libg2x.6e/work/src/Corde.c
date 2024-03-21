/*!=================================================================!*/
/*!= E.Incerti - eric.incerti@univ-eiffel.fr                       =!*/
/*!= Université Gustave Eiffel                                     =!*/
/*!= Code "squelette" pour prototypage avec libg2x.6e              =!*/
/*!= cet exemple se contente d'ouvrir une fenêtre graphique        =!*/
/*!=================================================================!*/

/* le seul #include nécessaire a priori
 * contient les libs C standards et OpenGl */
#include <g2x.h>
#include "PMat.h"
#include "Link.h"

/* -----------------------------------------------------------------------
 * ici, en général pas mal de variables GLOBALES
 * - les variables de données globales (points, vecteurs....)
 * - les FLAGS de dialogues
 * - les paramètres de dialogue
 * - ......
 * Pas trop le choix, puisque TOUT passe par des fonctions <void f(void)>
 * ----------------------------------------------------------------------- */

/* Paramètres de la fenêtre graphique */
static int pixwidth = 640, pixheight = 640;
static double xmin = -6., ymin = -6., xmax = +6., ymax = +6.;

/* Variables de simulation */
static double h; /* Pas d'échantillonnage 1./Fech */
static int Fe; /* Fréquence d'échantillonnage */
static double m, k, z; /* Paramètres physiques */
static char PouF = 'f'; /* Type de perturbation initiale */
static int tempo = 0; /* Temporisation de la simulation */

/* Système "Masses-Ressorts" : les particules et les liaisons */
static int nbm = 0;
static PMat *tabM = NULL;
static int nbl = 0;
static Link *tabL = NULL;

void Modeleur(void) {
    nbm = 10;
    if (!(tabM = (PMat *)calloc(nbm, sizeof(PMat))))
      g2x_Quit();
    nbl = nbm - 1;
    if (!(tabL = (Link *)calloc(nbm, sizeof(Link))))
      g2x_Quit();

    Fe = 100; // Paramètre du simulateur Fe = 1/h
    h = 1./Fe; 
    m = 1.; // La plupart du temps, toutes les masses sont à 1

    k = 0.01;  // raideurs -> à multiplier par Fe² (k * Fe² ~= 10.)
    z = 0.001; // viscosités -> à multiplier par Fe (z * Fe ~= 0.01)
                // k/z ~= 10 => (k * Fe²) / (z * Fe) ~= 1000

    /* les particules */
    PMat* M = tabM;
    M_builder(M++, 0, m, (G2Xpoint){0, 0}, g2x_Vector2d(0., 0.));
    for(int i = 1; i < nbm - 1; i++) {
        M_builder(M++, 1, m, (G2Xpoint){i, 0}, g2x_Vector2d(0., 0.));
    }
    M_builder(M++, 0, 0., (G2Xpoint){nbm-1, 0}, g2x_Vector2d(0., 0.));

    /* les liaisons */
    Link *L;
    for(L = tabL; L < tabL + nbl; L++) {
        /* param. k et z ajustés en fonction de Fe
         * k -> k * Fe²
         * z -> z * Fe
         * cf. <M2.Mot.Phys.2024-03.pdf> page 4
         * */
        Damped_Hook(L, k * SQR(Fe), z * Fe);
    }

    for((L = tabL, M = tabM); L < tabL + nbl; (L++, M++))
      Connect(M, L, M + 1);

    tabM[1].pos.y = 1;
}

/* la fonction d'initialisation : appelée 1 seule fois, au début (facultatif) */
static void init(void)
{
  /*!  Ici, pas de dessin, pas de calcul dynamique, rien que des initialisations
   *   -> allocation(s) de mémoire, init. des paramètres globaux,
   *      construction de modèles....
   *      ouverture de fichiers, pré-chargement de données....
   *
   *   Tout ce qu'il y a ici pourrait être écrit directement dans le main()
   *   juste après l'appel à g2x_InitWindow()
  !*/
  Modeleur();
}

/* la fonction de contrôle : appelée 1 seule fois, juste APRES <init> (facultatif) */
static void ctrl(void)
{
  /*! Interface de dialogue (partie statique) : création des boutons, scrollbars....
   *  Tout ce qu'il y a ici pourrait être directement écrit dans la fonction init(),
   *  mais c'est plus 'propre' et plus pratique de séparer.
  !*/
  g2x_CreateScrollv_d("h", &h, 1./Fe, (1./Fe)*5, "raideur");
  g2x_CreateScrollv_d("z", &z, 0, 10, "viscosité");
  g2x_CreateScrollv_d("force", &tabM[1].pos.y, 0, 4, "bouger un point pour donner de la force");
}

/* la fonction de contrôle : appelée 1 seule fois, juste APRES <init> (facultatif) */
static void evts(void)
{
  /*! Interface de dialogue (partie dynamique) - gestion des interruptions : clavier, souris ....
   *  Tout ce qu'il y a ici pourrait être directement écrit dans la fonction draw(),
   *  mais c'est plus 'propre' et plus pratique de séparer.
  !*/
}

/* la fonction d'animation : appelée en boucle draw/anim/draw/anim... (facultatif) */
static void anim(void)
{
  /*! C'est la fonction de 'calcul' qui va modifier les 'objets' affichés
   *  par la fonction de dessin (déplacement des objets, calculs divers...)
   *  Si elle n'est pas définie, c'est qu'il n'y a pas d'animation.
   *  ATTENTION : surtout pas d'alloc. mémoire ici !!!
  !*/
  for (Link* L = tabL; L < tabL + nbl; L++)
  {
    L->update(L);
    L->z = z;
  }
  for(PMat* M = tabM; M < tabM + nbm; M++)
  {
    M->update(M, h);
  }
}


/* la fonction de dessin : appelée en boucle (indispensable) */
static void draw(void)
{
  /*! C'est la fonction de dessin principale : elle ne réalise que de l'affichage
   *  sa mise à jour est automatique :
   *  - si un paramètre de contrôle est modifié
   *  - si la fonction <anim()> (calcul) est activée
   *  ATTENTION : surtout pas d'alloc. mémoire ici !!!
  !*/
  for (Link* L = tabL; L < tabL + nbl; L++)
  {
    if (L == tabL)
    {
      g2x_DrawFillCircle(g2x_Point2d(L->M1->pos.x, L->M1->pos.y), 0.5, G2Xy);
    }
    else if (L == tabL + nbl - 1)
    {
      g2x_DrawFillCircle(g2x_Point2d(L->M1->pos.x, L->M1->pos.y), 0.5, G2Xr);
      g2x_DrawFillCircle(g2x_Point2d(L->M2->pos.x, L->M2->pos.y), 0.5, G2Xy);
    }
    else
    {
      g2x_DrawFillCircle(g2x_Point2d(L->M1->pos.x, L->M1->pos.y), 0.5, G2Xr);
    }
    g2x_DrawLine(g2x_Point2d(L->M1->pos.x, L->M1->pos.y), g2x_Point2d(L->M2->pos.x, L->M2->pos.y), G2Xb, 1);
  }
}

/* la fonction de sortie  (facultatif) */
static void quit(void)
{
  /*! Ici, les opérations à réaliser en sortie du programme
   *  - libération de la mémoire éventuellement alloueé dans <init()>
   *  - fermeture de fichiers ....
   *  - bilan et messages...
   *  Au final cette fonction est exécutée par un appel à <atexit()>
  !*/
}

/***************************************************************************/
/* La fonction principale : NE CHANGE (presque) JAMAIS                     */
/***************************************************************************/
int main(int argc, char **argv)
{
  /** 1°) creation de la fenetre - titre et tailles (pixels) *
   *     tailles de la fenêtre graphique (en pixels)         *
   *     ces dimensions sont ajustables (redimensionnement)  *
   *     et accessibles via les fonctions de 'get'           *
   *     int g2x_GetPixWidth(); |  int g2x_GetPixHeight();   *
  **/
  int WWIDTH=512, WHEIGHT=512;
  g2x_InitWindow(*argv,WWIDTH,WHEIGHT);

  /** 2°) définition de la zone de travail en coord. réeelles   *
   *      limites de la zone reelle associee a la fenetre       *
   *      ATTENTION : ces valeurs doivent être compatibles avec *
   *      les tailles WWIDTH et WHEIGHT (rapport d'aspect 1:1)  *
   *      (wxmax-wxmin)/(wymax-wymin) = WWIDTH/WHEIGHT          *
   *    -> auto-ajustées en cas de redimensionnement fenêtre    *
   *    -> auto-ajustées en cas de zoom/panscan                 *
   *       et accessibles via les fonctions de 'get'            *
   *       int g2x_GetXMin(); |  int g2x_GetXMax(); ....        *
  **/
  double wxmin=-10.,wxmax=+10.,
         wymin=-10.,wymax=+10.;
  g2x_SetWindowCoord(wxmin,wymin,wxmax,wymax);

  /** 3°) association des fonctions **/
  g2x_SetInitFunction(init); /* fonction d'initialisation */
  g2x_SetCtrlFunction(ctrl); /* fonction de contrôle      */
  g2x_SetEvtsFunction(evts); /* fonction d'événements     */
  g2x_SetDrawFunction(draw); /* fonction de dessin        */
  g2x_SetAnimFunction(anim); /* fonction d'animation      */
  g2x_SetExitFunction(quit); /* fonction de sortie        */

  /** 4°) lancement de la boucle principale **/
  return g2x_MainStart();
  /** RIEN APRES CA **/
}