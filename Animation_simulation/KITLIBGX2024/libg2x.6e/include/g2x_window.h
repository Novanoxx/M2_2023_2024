
/**@file    g2x_window.h
 * @author  Universite Gustave Eiffel
 * @author  E.Incerti - eric.incerti@univ-eiffel.fr
 * @brief   Base public control functions
 * @version 6.e
 * @date    Aug.2022 (doc generation)
 */
/**@cond SHOW_SECURITY_DEFINE */

/*!==================================================================*/
#ifdef __cplusplus
  extern "C" {
#else
  #define _GNU_SOURCE
#endif

#ifndef _G2X_WINDOW_H_
  #define _G2X_WINDOW_H_
/**@endcond                   */

  #include <g2x.h>

  /*!****************************************************************************!*/
  /*! LA FENETRE GRAPHIQUE                                                       !*/
  /*!****************************************************************************!*/

  /*! initialisation de la fenetre et mise en place des parametres Gl
   * 1� fonction � appeler - indispensable
   * - windname : nom de la fen�tre (le + classique : argv[0])
   * - wwidth/wheight : tailles en pixel
  !*/
  void g2x_InitWindow(char* windname, int wwidth, int wheight);

  /*! definit la zone graphique reelle associee a la fenetre
   *  >> par d�faut c'est la zone [(0,0)->(wwidth,wheight)]
   *  >> ATTENTION : les ordonn�es (y) vont du BAS vers le HAUT
   *                 contrairement au coord. pixel standard
   *              -> Peut provoquer un renversement vertical
  !*/
  bool g2x_SetWindowCoord(double xmin, double ymin, double xmax, double ymax);

  /*!****************************************************************************!*/
  /*!  LES 6 FONCTIONS ESSENTIELLES DU MOTEUR GRAPHIQUE                          !*/
  /*!****************************************************************************!*/

  /*! 1�) INITIALISATION (chargement)
   *  >>> selectionne la fonction d'initialisation des donnees                        *
   *  >>> Cette fonction est appel�e une seule fois, avant le lancement
   *      de la boucle principale.
   *      C'est elle et elle seule qui doit contenir les �ventuelles allocation
   *      de m�moire.
   *      Elle se combine � la fonction de contr�le d�finie par <g2x_SetCtrlFunction>
   *  >>> Elle d�finit la partie "statique" du programme
   *  <<< Son usage n'est pas indispensable mais conseill�
   *      Tout son contenu pourrait �tre int�gr� directement dans la fonction <main>
   *
  !*/
  void g2x_SetInitFunction(void (*f)(void));

  /*! 2�) INTERFACE DE CONTR�LE (boutons, scrollbars, points de contr�le...)
   *  >>> selectionne la fonction des actions de contr�le (bouttons, scrollbars, clavier...)
   *  >>> Cette fonction est appel�e une seule fois, avant le lancement
   *      de la boucle principale, juste apr�s l'appel  � la fonction d'initialisation
   *      d�finie par <g2x_SetInitFunction>
   *  >>> Elle se combine aux actions et param�tres de contr�le pr�d�finis
   *  <<< Son usage n'est pas indispensable mais conseill�
   *      Tout son contenu pourrait �tre int�gr� dans la fonction d'initialisation
   *      ou directement dans la fonction <main>
  !*/
  void g2x_SetCtrlFunction(void (*f)(void));

  /*! 3�) GESTION DES EVENEMENTS (clavier, souris)
   *  >>> selectionne la fonction de gestion des �v�nements (clavier, souris...)
   *  >>> la gestion des interruption est volontairement brid�e et contraignante
   *      - seul le boutton gauche de la souris est disponible
   *        le boutton droit et la molette son associ�s � des action pr�d�finies
   *      - les touches clavier doivent �tre associ�es � des proc�dures autonomes
   *        de prototype <void action(void)> et certaines s�quences de touches sont
   *        verrouill�es pour des actions pr�d�finies [cf. <g2x_keys.h>] :
   *        <ESC>,<SPACE>,<F11>,<F12>
   *        <Ctrl+'q'>,<Ctl+'f'>,<Ctrl+'w'>,<Ctrl+'r'>
   *        '?','+','-','='
   *      - la prise en compte des interruptions est synchronis�e avec l'appel
   *        � la fonction de 'display' (ou 'refresh')
  !*/
  void g2x_SetEvtsFunction(void (*f)(void));

  /*! 4�) AFFICHAGE - FONCTION DE DESSIN
   *  >>> selectionne la fonction associee aux actions de d'affichage (Display)
   *  >>> elle est indispensable pour afficher quelque chose
   *  >>> elle est appel�e en boucle infinie, en synchro avec la fonction d'anim.
   *      - elle ne doit contenir QUE des appels d'affichage
   *      - SURTOUT PAS d'appel de gestion m�moire (malloc/free)
   *  >>> la synchro affichage/calcul est g�r�e par le moteur GL mais peut �tre
   *      param�tr� par la fonction <g2x_SetRefreshFreq(int F_AFF)> qui fixe le
   *      rapport de fr�quence d'appel pour ces 2 fctions
   *      <F_AFF> appels � la fonction d'anim pour 1 appel � la fonction d'affichage
  !*/
  void g2x_SetDrawFunction(void (*f)(void));

  /*! 5�) ANIMATION - FONCTION DE CALCUL
   *  >>> selectionne la fonction associee aux actions d'animation / calculs
   *  >>> elle est indispensable pour animer quelque chose
   *  >>> elle est appel�e en boucle infinie, en synchro avec la fonction d'affichage.
   *      - elle ne doit contenir QUE des calculs
   *      - SURTOUT PAS d'appel de gestion m�moire (malloc/free)
   *      - pas d'appel d'affichage (ne serait pa pris en compte).
   *  >>> la synchro affichage/calcul est g�r�e par le moteur GL mais peut �tre
   *      param�tr� par la fonction <g2x_SetRefreshFreq(int F_AFF)> qui fixe le
   *      rapport de fr�quence d'appel pour ces 2 fctions
   *      <F_AFF> appels � la fonction d'anim pour 1 appel � la fonction d'affichage
  !*/
  void g2x_SetAnimFunction(void (*f)(void));

  /*! 6�) SORTIE
   *  >>> selectionne la fonction associee aux actions � faire en sortie (atexit)
   *  >>> c'est elle, en particulier, qui doit lib�rer la m�moire �ventuellement
   *      allou�e dans la fonction d'initialisation
   *  >>> elle est appel�e une seule fois, � la sortie du programme
  !*/
  void g2x_SetExitFunction(void (*f)(void));

  /*!****************************************************************************!*/
  /*!  LA FONCTION DE DEMARRAGE                                                  !*/
  /*!****************************************************************************!*/
  /*! lance la boucle principale (glutMainLoop)
   *  >>> derni�re fonction � appeler
  !*/
  int  g2x_MainStart(void);


  /*!****************************************************************************!*/
  /*!  RECUPERATION DES VARIABLES D'ENVIRONNEMENT                                !*/
  /*!****************************************************************************!*/

  /*! affichage/anim : pause de <usec> micro-sec. entre chaque appel � <anim>    !*/
  void   g2x_SetDelay(uint usec);

  /*! affichage/anim : 1 affichage pour <freq> passes de calcul                  !*/
  void   g2x_SetRefreshFreq(int freq);


  /*! les tailles de l'�cran (pixels) */
  int    g2x_GetScreenWidth (void);
  int    g2x_GetScreenHeight(void);

  /*! les tailles de la fen�tre graphique (pixels)
   *  >>> elles changent lorsque la fen�tre est redimensionn�e
  !*/
  int    g2x_GetPixWidth(void);  /* largeur courante (en pixels)                  */
  int    g2x_GetPixHeight(void); /* hauteur courante (en pixels)                  */

  /*! les limites de la fen�tre r�elle
   *  Elle change (ou pas...) lorsque la fen�tre graphique est redimensionn�e
  !*/
  double g2x_GetXMin(void);      /* coordonnee courante en X du bord gauche       */
  double g2x_GetYMin(void);      /* coordonnee courante en Y du bord inf�rieur    */
  double g2x_GetXMax(void);      /* coordonnee courante en X du bord droit        */
  double g2x_GetYMax(void);      /* coordonnee courante en Y du bord sup�rieur    */

  double g2x_GetXPixSize(void);  /* taille reelle du pixel en X                   */
  double g2x_GetYPixSize(void);  /* taille reelle du pixel en Y                   */

  /*! r�glage/r�cup�ration du facteur de zoom !*/
  void   g2x_SetZoom(double z);  /* fixe  la valeur du acteur de Zoom (x,y)       */
  double g2x_GetZoom(void);      /* r�cup�re la valeur du acteur de Zoom (x,y)    */

  /*! fixe / recupere la couleur de fond de la fenetre                            !*/
  void   g2x_SetBkGdCol(double BkGd);
  double g2x_GetBkGdCol(void      );


  /*!****************************************************************************!*/
  /*!  UTILITAIRES PRATIQUES                                                     !*/
  /*!****************************************************************************!*/

  /*! TRUE par d�faut - pour d�activer Zoom et Panscan (fige la fen�tre)          !*/
  void   g2x_SetPanScanAndZoom(bool FLAG);
  /*! trace les axes (0,x,y)                                                      !*/
  void   g2x_Axes(void);
  /*! trace une grilles de graduation                                             !*/
  void   g2x_Grad(G2Xcolor col_axe, double step, G2Xcolor col_grad);
  /*! trace une grilles de graduation � plusieurs niveaux                         !*/
  void   g2x_MultiGrad(G2Xcolor col_axe, int nb_level, ...);

  /*! ====================================================================== !*/
  /*! Affichage de texte
   *  >> OpenGl est tr�s limit� en ce qui concerne l'affichage de texte
   *     il ne propose que 2 fontes, 4 tailles et 1 seul style (normal)
   *     -> n�anmoins un style 'bold' (gras) existe ici
   *     De plus l'affichage de texte s'av�re assez co�teux.
   *     -> donc � �viter dans les appli visant la performance (animations)
  !*/

  /*! Selection de fonte d'�criture (taille et style) :
   *  size  's' (small) 'm' (medium) 'l' (large) 'L' (Large)
   *  style 'n' (normal) 'b' (bold)
   *  pos   'l','c','r' : left, center, right
   *  > la valeur 0 laisse le param�tre inchang� :
   *  > g2x_SetFontAttributes(0,'b',0) : passe en style <bold> sans changer la taille et la position.
  !*/
  void g2x_SetFontAttributes(char size, char style, char pos);

  /*! affiche en coord. 'pixel' (entiers : x,y)
   *  les attributs (taille, style, position) sont fix�s par <g2x_SetFontAttributes>
   *  col : couleur du texte
   *  le reste : comme <printf(const char *format, ...)>
   *  le texte garde sa position dans la fen�tre ind�pendemment du redimensionnement,
   *  et des variations (zoom/panscan) de la fen�tre r�elle.
  !*/
  void g2x_StaticPrint(int x, int y, G2Xcolor col, const char *format, ...);

  /*! affiche en coord. 'fenetre' (r�eels : x,y)
   *  les attributs (taille, style, position) sont fix�s par <g2x_SetFontAttributes>
   *  col : couleur du texte
   *  le reste : comme <printf(const char *format, ...)> *
   *  le texte suit les variations (zoom/panscan) de la fen�tre r�elle.
  !*/
  void g2x_Print(double x, double y, G2Xcolor col, const char *format, ...);


  /*! comme <g2x_StaticPrint> mais avec un cadre
   *  (x,y)  : position d'ancrage fen�tre
   *  margin : marge entre le texte et le cadre
   *  txtcol : couleur du texte et du cadre
   *  bkgcol : couleur de fond
   *  borderw: epaisseur du cadre (pixels)
   *  le reste : comme <printf(const char *format, ...)> *
  !*/
  void g2x_StaticTextBox(int x, int y, int margin,
                         G2Xcolor txtcol, G2Xcolor bkgcol, int borderw,
                         const char *format, ...);

  /*!****************************************************************************!*/
  /*!  UTILITAIRES D'USAGE PLUS MARGINAL                                         !*/
  /*!****************************************************************************!*/

  /*! int�gr� � la fin de la fonction d'affichage (pass�e � <g2x_SetDrawFunction>,
   *  cet appel permet de faire de l'image par image dans une animation
   *  => affiche une image et s'arr�te.
   *  => la touche "espace" permet de passer � l'�tape suivante. !*/
  void   g2x_Stop(void);

  /*! relance l'animation (cas d'usage tr�s rare)        !*/
  void   g2x_Continue(void);

  /*! renvoie <true> si la boucle d'animation est active !*/
  bool   g2x_Running(void);

  /*! force l'affichage (usage limit� � quelques situations un peu sp�ciales)    !*/
  void   g2x_Refresh(void);

  /*! pour sortir proprement en cas d'urgence (erreur)
   *  >>> elle int�gre la fonction de sortie utilisateur <g2x_SetExitFunction>
   *  >>> normalement il n'y a aucune raison d'en faire usage !*/
  void g2x_Quit(void);

 #endif

#ifdef __cplusplus
  }
#endif
/*!=============================================================!*/
