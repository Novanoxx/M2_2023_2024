#ifndef _PMAT_
    #define _PMAT_
    #include <g2x.h>

    typedef struct _ptm_
    {
        float m; //masse
        G2Xpoint pos; //position
        G2Xvector vit; //vitesse
        G2Xvector frc; //buffer d’accumulation des forces
        //mise à jour : intégrateur (h: pas de calcul)
        void (*update)(struct _ptm_ *this, float h);
    } PMat;
    // Constructeur
    void M_builder(PMat *M, int type, float m, G2Xpoint P0, G2Xvector V0);
#endif