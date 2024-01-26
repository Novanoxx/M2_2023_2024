#include <g2x.h>

typedef struct _ptm_
{
    float m; //masse
    G2Xpoint pos; //position
    double vit; //vitesse
    double frc; //buffer d’accumulation des forces
    //mise à jour : intégrateur (h: pas de calcul)
    void (*update)(struct _ptm_ *this, float h);
} PMat;
// Constructeur
void M_builder(PMat *M, int type, float m, G2Xpoint P0, double V0);