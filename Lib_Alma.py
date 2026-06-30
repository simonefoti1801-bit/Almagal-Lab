import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import radians, cos, sin
import pandas as pd
import astropy as ap
import fitsio as fi
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS

#Useful functions used in the main codes
def deg_to_rad(theta):
    return theta * np.pi/180
def rad_to_arcsec(theta):
    return theta * ( 180 * 3600)/(np.pi)
def deg_to_arcsec(theta):
    return theta * 3600
def read_txt(filename_txt):
    df_fake = pd.read_csv(filename_txt, sep='\s+',skiprows=1 ,dtype={'CLUMP': str})
    df = df_fake[['ID','CLUMP','RA','DEC','FWHM_X','FWHM_Y','PA','FWHM_circ','Lclump','Mclump','ID_ord','Tclump','RMS_map','Dclump','Surfd']]
    df = df.iloc[1:].reset_index(drop=True) #toglie la prima riga e resetta gli indici
    numeric_cols = ['ID','RA', 'DEC', 'FWHM_X', 'FWHM_Y', 'PA', 'FWHM_circ','Lclump','Mclump','ID_ord','Tclump','RMS_map','Dclump','Surfd']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce') #trasforma il dataframe in cui potrebbero esserci stringhe, in tutti numeri. LASCIO CLUMP IN STRINGHE PERCHE' ALCUNI HANNO NOMI STRANI!
    return df
#!!!Insert in catalogue name the path where the catalogues are stored!!!
def catalogue_name(configuration):
    return '/Users/simonefoti/Desktop/Almagal/2°Parte/Catalogs/'+configuration+'_Core_catalogue.txt'

#First main function that creates the family tree of the sources in the three catalogues. It returns the family_clump and family_id 3d tensor, and the orphans matrix for B and array for C.
def family_tree(n,m,N=20,M=20):
   #In this function the 7M catalogue is defined A, the 7MTM2 B, the 7MTM2TM1 C
    #n is the threshold for the sources to be son of in unity of FWHM from A->B
    #m is the threshold for the sources to be son of in unity of FWHM from B->C
    #N is the length of the x-axis on the histogram. 30 is enough for sigma<=3

    df_A=read_txt(catalogue_name('7M'))
    CLUMP_A=df_A['CLUMP'].to_numpy()
    ID_A=df_A['ID'].to_numpy()
    RA_A=rad_to_arcsec( deg_to_rad(df_A['RA'].to_numpy() ))
    DEC_A=rad_to_arcsec(deg_to_rad(df_A['DEC'].to_numpy()))
    FWHM_X_A=df_A['FWHM_X'].to_numpy()
    FWHM_Y_A=df_A['FWHM_Y'].to_numpy()
    FWHM_circ_A=df_A['FWHM_circ'].to_numpy()
    PA_A=df_A['PA'].to_numpy()

    df_B=read_txt(catalogue_name('7MTM2'))
    CLUMP_B=df_B['CLUMP'].to_numpy()
    ID_B=df_B['ID'].to_numpy()
    RA_B=rad_to_arcsec( deg_to_rad(df_B['RA'].to_numpy() ))
    DEC_B=rad_to_arcsec(deg_to_rad(df_B['DEC'].to_numpy()))
    FWHM_X_B=df_B['FWHM_X'].to_numpy()
    FWHM_Y_B=df_B['FWHM_Y'].to_numpy()
    FWHM_circ_B=df_B['FWHM_circ'].to_numpy()
    PA_B=df_B['PA'].to_numpy()

    df_C=read_txt(catalogue_name('7MTM2TM1'))
    CLUMP_C=df_C['CLUMP'].to_numpy()
    ID_C=df_C['ID'].to_numpy()
    RA_C=rad_to_arcsec( deg_to_rad(df_C['RA'].to_numpy() ))
    DEC_C=rad_to_arcsec(deg_to_rad(df_C['DEC'].to_numpy()))
    FWHM_X_C=df_C['FWHM_X'].to_numpy()
    FWHM_Y_C=df_C['FWHM_Y'].to_numpy()
    FWHM_circ_C=df_C['FWHM_circ'].to_numpy()
    PA_C=df_C['PA'].to_numpy()

    
    family_clump = np.empty((len(CLUMP_A), N, M), dtype=object)
    family_id = np.empty((len(CLUMP_A), N, M), dtype=object)

    b_with_a_father=np.zeros(len(CLUMP_B), dtype=bool)
    c_with_a_father=np.zeros(len(CLUMP_C),dtype=bool)
    # 1. Trova legami B -> C prima del ciclo principale e salva in un dizionario
    links_BC = {} 
   
    for j in range(len(CLUMP_B)):
        links_BC[j] = []
        for k in range(len(CLUMP_C)):
            dist = ((RA_B[j]-RA_C[k])**2 + (DEC_B[j]-DEC_C[k])**2)**0.5
            if dist <= m * FWHM_circ_B[j] and df_B['CLUMP'].iloc[j]==df_C['CLUMP'].iloc[k]: #qui uso i dataframe perchè mi servono i clump come stringhe per confrontare quelli con le G
                links_BC[j].append( (str(CLUMP_C[k]), str(ID_C[k])) ) 
                c_with_a_father[k]=1
                
    

    for i in range(len(CLUMP_A)):
        family_clump[i][0][0]=str(CLUMP_A[i])
        family_id[i][0][0]=str(ID_A[i])
        c=0
        for j in range(len(CLUMP_B)):
            l=0
            dist =  (  ( RA_A[i]-RA_B[j] ) **2 +   (DEC_A[i]-DEC_B[j] ) **2 )**0.5
            if (   dist <=  n * FWHM_circ_A[i]   and df_A['CLUMP'].iloc[i]==df_B['CLUMP'].iloc[j]) :
                c+=1 
                b_with_a_father[j]=1
                if c < N:
                    clump_B = df_B['CLUMP'].iloc[j]
                    id_B=str(ID_B[j])
                    family_clump[i][c][0] = clump_B
                    family_id[i][c][0]= id_B
                    # in name_C ho gli elementi del dizionario, m è l'indice di elemento. Quindi scorro nell'
                    for m, (clump_C, id_C) in enumerate(links_BC[j]):
                        if m + 1 < M:
                            family_clump[i][c][m+1] = clump_C
                            family_id[i][c][m+1] = id_C

    orphan_indices_b = np.where(b_with_a_father== False)[0]#ritorna un array con gli indici delle righe del catalogo b di sorgenti che non sono figlie di A
    orphans_clump_b=np.empty((len(orphan_indices_b), M), dtype=object)
    orphans_id_b=np.empty((len(orphan_indices_b), M), dtype=object)
    #QUESTO FOR CARICA NEGLI ELEMENTI  i0 DELLA MATRICE LE SORGENTI IN B NON FIGLIE DI A, E NELLA RIGA ij LE FIGLIE IN C DI B
    for idx, j in enumerate(orphan_indices_b): #faccio for su j che scorre negli elementi di orphan_indices, index è una nuova variabile che contiene il numero del ciclo di questo for
        orphans_clump_b[idx, 0] = df_B['CLUMP'].iloc[j]
        orphans_id_b[idx, 0] = str(ID_B[j])
        # Aggiungiamo anche qui le figlie C usando il dizionario pre-calcolato
        for m, (clump_C, id_C) in enumerate(links_BC[j]):
            if m + 1 < M:
                orphans_clump_b[idx, m+1] = clump_C
                orphans_id_b[idx, m+1] = id_C

    orphan_indices_c=np.where(c_with_a_father==False)[0]
    orphans_clump_c=np.zeros(len(orphan_indices_c),dtype=object)
    orphans_id_c=np.zeros(len(orphan_indices_c),dtype=object)
    #QUESTO FOR CARICA NEGLI ARRAY LE SORGENTI IN C NON FIGLIE DI B 
    for idx,j in enumerate(orphan_indices_c):
        orphans_clump_c[idx]=df_C['CLUMP'].iloc[j]
        orphans_id_c[idx]=str(ID_C[j])

                                
    return family_clump,family_id,orphans_clump_b,orphans_id_b,orphans_clump_c,orphans_id_c
#Functions for the report generation. The first one draws the ellipses and the others generate the report for the clumps. The first one without reprojection, the second one with reprojection of the first two catalogues on the third one.
def draw_ellipse(ax, wcs, df_indexed, clump, src_id, color, role):
    try:
        # Pulisce clump e id da ogni possibile formato (float, int, stringa)
        # Trasformare in float e poi int rimuove il ".0" fastidioso delle stringhe
        c_key = str(clump).strip()
        try:
            i_key = str(int(float(src_id))).strip()
        except:
            i_key = str(src_id).strip()

        if i_key in ['None', 'nan', '', '0']: return

        # Recupero riga
        if (c_key, i_key) not in df_indexed.index:
            # print(f"Salto: {c_key}-{i_key} non in indice") # Debug opzionale
            return

        row = df_indexed.loc[(c_key, i_key)]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        
        # Centro (RA/DEC in gradi)
        px, py = wcs.all_world2pix(row['RA'], row['DEC'], 0)
        
        # Dimensioni (FWHM in arcsec -> pixel)
        pix_scale = np.mean(np.abs(wcs.wcs.cdelt)) * 3600.0 
        w_px = row['FWHM_X'] / pix_scale
        h_px = row['FWHM_Y'] / pix_scale
        
        # Disegno
        ell = Ellipse((px, py), w_px, h_px, angle=row['PA'], 
                      edgecolor=color, facecolor='none', lw=0.5, zorder=10)
        ax.add_artist(ell)
        
        # Etichetta
        label = "O" if role == 'O' else i_key
        ax.text(px+10, py+10, label, color=color, fontsize=7, ha='center', 
                va='center', fontweight='bold', zorder=11)
        
    except Exception as e:
        # print(f"Errore critico su {clump}-{src_id}: {e}")
        pass
def genera_report_clumps_finale(
    output_pdf,
    family_clump,
    family_id,
    orphans_clump_b,
    orphans_id_b,
    orphans_clump_c,
    orphans_id_c,
    df_A,
    df_B,
    df_C,
    n_clumps=None
):
#prendere header primo e usarlo per riproiettare dati astrometrici secondo e terzo
    import time
    import warnings

    warnings.filterwarnings("ignore")

    # =========================
    # LISTA CLUMPS
    # =========================
    tutti_i_clumps = np.unique(df_A['CLUMP'].to_numpy())

    if n_clumps is not None:
        tutti_i_clumps = tutti_i_clumps[:n_clumps]

    totale_clumps = len(tutti_i_clumps)

    print(f"\nGenerazione report per {totale_clumps} clumps...\n")

    clump_to_fits_id = dict(zip(df_A['CLUMP'], df_A['ID_ord']))

    # =========================
    # CATALOGHI INDICIZZATI
    # =========================
    cats = {
        '7M': df_A.assign(
            CLUMP=df_A['CLUMP'].astype(str),
            ID=df_A['ID'].astype(str)
        ).set_index(['CLUMP', 'ID']),

        '7MTM2': df_B.assign(
            CLUMP=df_B['CLUMP'].astype(str),
            ID=df_B['ID'].astype(str)
        ).set_index(['CLUMP', 'ID']),

        '7MTM2TM1': df_C.assign(
            CLUMP=df_C['CLUMP'].astype(str),
            ID=df_C['ID'].astype(str)
        ).set_index(['CLUMP', 'ID'])
    }

    colors = plt.cm.tab20.colors

    start_time = time.time()

    # =========================
    # PDF
    # =========================
    with PdfPages(output_pdf) as pdf:

        clump_counter = 0

        for p in range(0, len(tutti_i_clumps), 3):

            clumps_pagina = tutti_i_clumps[p:p+3]

            fig, axes = plt.subplots(
                len(clumps_pagina),
                3,
                figsize=(18, 6 * len(clumps_pagina))
            )

            if len(clumps_pagina) == 1:
                axes = np.expand_dims(axes, axis=0)

            # ==========================================================
            # LOOP SUI CLUMPS
            # ==========================================================
            for row, clump_name in enumerate(clumps_pagina):

                clump_counter += 1

                percentuale = 100 * clump_counter / totale_clumps

                elapsed = time.time() - start_time

                print(
                    f"[{clump_counter}/{totale_clumps}] "
                    f"{percentuale:5.1f}%  "
                    f"CLUMP: {clump_name}   "
                    f"Tempo: {elapsed:6.1f}s"
                )

                suffissi = ['7M', '7MTM2', '7MTM2TM1']

                num_id = clump_to_fits_id.get(clump_name)

                for col, cat_name in enumerate(suffissi):

                    ax = axes[row, col]

                    if num_id is None:
                        ax.text(
                            0.5,
                            0.5,
                            f"ID non trovato\n{clump_name}",
                            ha='center'
                        )
                        continue

                    nome_fits = (
                        f"{str(clump_name).strip()}_cont_"
                        f"{cat_name}_jointdeconv.image.fits"
                    )

                    try:

                        with fits.open(nome_fits) as hdul:

                            data = hdul[0].data

                            if data.ndim == 4:
                                data = data[0, 0, :, :]

                            elif data.ndim == 3:
                                data = data[0, :, :]
                            ####
                            wcs = WCS(hdul[0].header).celestial

                            ax.imshow(
                                data,
                                origin='lower',
                                cmap='magma',
                                interpolation='nearest'
                            )

                            ax.set_title(
                                f"CLUMP: {clump_name}\nCATALOGO: {cat_name}",
                                fontsize=10,
                                fontweight='bold',
                                pad=10
                            )

                            idx_A = np.where(
                                family_clump[:, 0, 0] == clump_name
                            )[0]

                            # ==================================================
                            # 7M
                            # ==================================================
                            if cat_name == '7M':

                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7M'],
                                        clump_name,
                                        family_id[i, 0, 0],
                                        'cyan',
                                        p_id
                                    )

                            # ==================================================
                            # 7MTM2
                            # ==================================================
                            elif cat_name == '7MTM2':

                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    contatore_figlie = 1

                                    for c in range(1, family_clump.shape[1]):

                                        if family_clump[i, c, 0] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            colore_ramo = colors[c % len(colors)]

                                            label_figlia = (
                                                f"{p_id}.{contatore_figlie}"
                                            )

                                            draw_ellipse(
                                                ax,
                                                wcs,
                                                cats['7MTM2'],
                                                clump_name,
                                                family_id[i, c, 0],
                                                colore_ramo,
                                                label_figlia
                                            )

                                            contatore_figlie += 1

                                # ==========================================
                                # ORFANE B
                                # ==========================================
                                idx_O = np.where(
                                    orphans_clump_b[:, 0] == clump_name
                                )[0]

                                for j in idx_O:

                                    o_id = str(
                                        int(float(orphans_id_b[j, 0]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7MTM2'],
                                        clump_name,
                                        orphans_id_b[j, 0],
                                        'red',
                                        f"O_{o_id}"
                                    )

                            # ==================================================
                            # 7MTM2TM1
                            # ==================================================
                            elif cat_name == '7MTM2TM1':

                                # ==========================================
                                # NIPOTI NORMALI
                                # ==========================================
                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    contatore_figlie = 1

                                    for c in range(1, family_clump.shape[1]):

                                        if family_clump[i, c, 0] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            colore_ramo = colors[c % len(colors)]

                                            contatore_nipoti = 1

                                            for m in range(
                                                1,
                                                family_clump.shape[2]
                                            ):

                                                if family_clump[i, c, m] not in [
                                                    'None',
                                                    'nan',
                                                    '',
                                                    None
                                                ]:

                                                    label_nipote = (
                                                        f"{p_id}."
                                                        f"{contatore_figlie}."
                                                        f"{contatore_nipoti}"
                                                    )

                                                    draw_ellipse(
                                                        ax,
                                                        wcs,
                                                        cats['7MTM2TM1'],
                                                        clump_name,
                                                        family_id[i, c, m],
                                                        colore_ramo,
                                                        label_nipote
                                                    )

                                                    contatore_nipoti += 1

                                            contatore_figlie += 1

                                # ==========================================
                                # FIGLIE IN C DELLE ORFANE B
                                # ==========================================
                                idx_OB = np.where(
                                    orphans_clump_b[:, 0] == clump_name
                                )[0]

                                for j in idx_OB:

                                    o_id = str(
                                        int(float(orphans_id_b[j, 0]))
                                    )

                                    contatore_figlie_orfane = 1

                                    for m in range(
                                        1,
                                        orphans_clump_b.shape[1]
                                    ):

                                        if orphans_clump_b[j, m] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            label_orphan_child = (
                                                f"O_{o_id}."
                                                f"{contatore_figlie_orfane}"
                                            )

                                            draw_ellipse(
                                                ax,
                                                wcs,
                                                cats['7MTM2TM1'],
                                                clump_name,
                                                orphans_id_b[j, m],
                                                'red',
                                                label_orphan_child
                                            )

                                            contatore_figlie_orfane += 1

                                # ==========================================
                                # ORFANE PURE IN C
                                # ==========================================
                                idx_OC = np.where(
                                    orphans_clump_c == clump_name
                                )[0]

                                for j in idx_OC:

                                    oc_id = str(
                                        int(float(orphans_id_c[j]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7MTM2TM1'],
                                        clump_name,
                                        orphans_id_c[j],
                                        'white',
                                        f"OC_{oc_id}"
                                    )

                            ax.axis('off')

                    except FileNotFoundError:

                        ax.text(
                            0.5,
                            0.5,
                            f"MANCANTE:\n{nome_fits}",
                            ha='center',
                            color='red'
                        )

                        ax.set_title(
                            f"Clump: {clump_name} | {cat_name}",
                            color='gray'
                        )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

            plt.close(fig)

    total_time = time.time() - start_time

    print(f"\nPDF Generato: {output_pdf}")
    print(f"Tempo totale: {total_time:.1f} s\n")
def genera_report_clumps_finale_proj(
    output_pdf,
    family_clump,
    family_id,
    orphans_clump_b,
    orphans_id_b,
    orphans_clump_c,
    orphans_id_c,
    df_A,
    df_B,
    df_C,
    n_clumps=None
):

    import time
    import warnings

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.backends.backend_pdf import PdfPages

    from astropy.io import fits
    from astropy.wcs import WCS

    from reproject import reproject_interp

    warnings.filterwarnings("ignore")

    # =========================
    # LISTA CLUMPS
    # =========================
    tutti_i_clumps = np.unique(df_A['CLUMP'].to_numpy())

    if n_clumps is not None:
        tutti_i_clumps = tutti_i_clumps[:n_clumps]

    totale_clumps = len(tutti_i_clumps)

    print(f"\nGenerazione report per {totale_clumps} clumps...\n")

    clump_to_fits_id = dict(zip(df_A['CLUMP'], df_A['ID_ord']))

    # =========================
    # CATALOGHI INDICIZZATI
    # =========================
    cats = {
        '7M': df_A.assign(
            CLUMP=df_A['CLUMP'].astype(str),
            ID=df_A['ID'].astype(str)
        ).set_index(['CLUMP', 'ID']),

        '7MTM2': df_B.assign(
            CLUMP=df_B['CLUMP'].astype(str),
            ID=df_B['ID'].astype(str)
        ).set_index(['CLUMP', 'ID']),

        '7MTM2TM1': df_C.assign(
            CLUMP=df_C['CLUMP'].astype(str),
            ID=df_C['ID'].astype(str)
        ).set_index(['CLUMP', 'ID'])
    }

    colors = plt.cm.tab20.colors

    start_time = time.time()

    # =========================
    # PDF
    # =========================
    with PdfPages(output_pdf) as pdf:

        clump_counter = 0

        for p in range(0, len(tutti_i_clumps), 3):

            clumps_pagina = tutti_i_clumps[p:p+3]

            fig, axes = plt.subplots(
                len(clumps_pagina),
                3,
                figsize=(18, 6 * len(clumps_pagina))
            )

            if len(clumps_pagina) == 1:
                axes = np.expand_dims(axes, axis=0)

            # ==========================================================
            # LOOP SUI CLUMPS
            # ==========================================================
            for row, clump_name in enumerate(clumps_pagina):

                clump_counter += 1

                percentuale = 100 * clump_counter / totale_clumps

                elapsed = time.time() - start_time

                print(
                    f"[{clump_counter}/{totale_clumps}] "
                    f"{percentuale:5.1f}%  "
                    f"CLUMP: {clump_name}   "
                    f"Tempo: {elapsed:6.1f}s"
                )

                suffissi = ['7M', '7MTM2', '7MTM2TM1']

                num_id = clump_to_fits_id.get(clump_name)

                # ======================================================
                # FITS DI RIFERIMENTO (7MTM2TM1)
                # ======================================================
                nome_fits_ref = (
                    f"{str(clump_name).strip()}_cont_"
                    f"7MTM2TM1_jointdeconv.image.fits"
                )

                try:

                    with fits.open(nome_fits_ref) as hdul_ref:

                        data_ref = hdul_ref[0].data

                        if data_ref.ndim == 4:
                            data_ref = data_ref[0, 0, :, :]

                        elif data_ref.ndim == 3:
                            data_ref = data_ref[0, :, :]

                        wcs_ref = WCS(hdul_ref[0].header).celestial

                        shape_ref = data_ref.shape

                except FileNotFoundError:

                    for col in range(3):

                        ax = axes[row, col]

                        ax.text(
                            0.5,
                            0.5,
                            f"MANCANTE:\n{nome_fits_ref}",
                            ha='center',
                            color='red'
                        )

                        ax.axis('off')

                    continue

                for col, cat_name in enumerate(suffissi):

                    ax = axes[row, col]

                    if num_id is None:

                        ax.text(
                            0.5,
                            0.5,
                            f"ID non trovato\n{clump_name}",
                            ha='center'
                        )

                        continue

                    nome_fits = (
                        f"{str(clump_name).strip()}_cont_"
                        f"{cat_name}_jointdeconv.image.fits"
                    )

                    try:

                        with fits.open(nome_fits) as hdul:

                            data = hdul[0].data

                            if data.ndim == 4:
                                data = data[0, 0, :, :]

                            elif data.ndim == 3:
                                data = data[0, :, :]

                            wcs_current = WCS(
                                hdul[0].header
                            ).celestial

                            # ==========================================
                            # RIPROIEZIONE SU WCS DEL TERZO CATALOGO
                            # ==========================================
                            if cat_name != '7MTM2TM1':

                                data_reproj, _ = reproject_interp(
                                    (data, wcs_current),
                                    wcs_ref,
                                    shape_out=shape_ref
                                )

                                data = data_reproj
                                wcs = wcs_ref

                            else:

                                data = data_ref
                                wcs = wcs_ref

                            ax.imshow(
                                data,
                                origin='lower',
                                cmap='magma',
                                interpolation='nearest'
                            )

                            ax.set_title(
                                f"CLUMP: {clump_name}\nCATALOGO: {cat_name}",
                                fontsize=10,
                                fontweight='bold',
                                pad=10
                            )

                            idx_A = np.where(
                                family_clump[:, 0, 0] == clump_name
                            )[0]

                            # ==================================================
                            # 7M
                            # ==================================================
                            if cat_name == '7M':

                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7M'],
                                        clump_name,
                                        family_id[i, 0, 0],
                                        'cyan',
                                        p_id
                                    )

                            # ==================================================
                            # 7MTM2
                            # ==================================================
                            elif cat_name == '7MTM2':

                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    contatore_figlie = 1

                                    for c in range(1, family_clump.shape[1]):

                                        if family_clump[i, c, 0] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            colore_ramo = colors[c % len(colors)]

                                            label_figlia = (
                                                f"{p_id}.{contatore_figlie}"
                                            )

                                            draw_ellipse(
                                                ax,
                                                wcs,
                                                cats['7MTM2'],
                                                clump_name,
                                                family_id[i, c, 0],
                                                colore_ramo,
                                                label_figlia
                                            )

                                            contatore_figlie += 1

                                # ==========================================
                                # ORFANE B
                                # ==========================================
                                idx_O = np.where(
                                    orphans_clump_b[:, 0] == clump_name
                                )[0]

                                for j in idx_O:

                                    o_id = str(
                                        int(float(orphans_id_b[j, 0]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7MTM2'],
                                        clump_name,
                                        orphans_id_b[j, 0],
                                        'red',
                                        f"O_{o_id}"
                                    )

                            # ==================================================
                            # 7MTM2TM1
                            # ==================================================
                            elif cat_name == '7MTM2TM1':

                                # ==========================================
                                # NIPOTI NORMALI
                                # ==========================================
                                for i in idx_A:

                                    p_id = str(
                                        int(float(family_id[i, 0, 0]))
                                    )

                                    contatore_figlie = 1

                                    for c in range(1, family_clump.shape[1]):

                                        if family_clump[i, c, 0] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            colore_ramo = colors[c % len(colors)]

                                            contatore_nipoti = 1

                                            for m in range(
                                                1,
                                                family_clump.shape[2]
                                            ):

                                                if family_clump[i, c, m] not in [
                                                    'None',
                                                    'nan',
                                                    '',
                                                    None
                                                ]:

                                                    label_nipote = (
                                                        f"{p_id}."
                                                        f"{contatore_figlie}."
                                                        f"{contatore_nipoti}"
                                                    )

                                                    draw_ellipse(
                                                        ax,
                                                        wcs,
                                                        cats['7MTM2TM1'],
                                                        clump_name,
                                                        family_id[i, c, m],
                                                        colore_ramo,
                                                        label_nipote
                                                    )

                                                    contatore_nipoti += 1

                                            contatore_figlie += 1

                                # ==========================================
                                # FIGLIE IN C DELLE ORFANE B
                                # ==========================================
                                idx_OB = np.where(
                                    orphans_clump_b[:, 0] == clump_name
                                )[0]

                                for j in idx_OB:

                                    o_id = str(
                                        int(float(orphans_id_b[j, 0]))
                                    )

                                    contatore_figlie_orfane = 1

                                    for m in range(
                                        1,
                                        orphans_clump_b.shape[1]
                                    ):

                                        if orphans_clump_b[j, m] not in [
                                            'None',
                                            'nan',
                                            '',
                                            None
                                        ]:

                                            label_orphan_child = (
                                                f"O_{o_id}."
                                                f"{contatore_figlie_orfane}"
                                            )

                                            draw_ellipse(
                                                ax,
                                                wcs,
                                                cats['7MTM2TM1'],
                                                clump_name,
                                                orphans_id_b[j, m],
                                                'red',
                                                label_orphan_child
                                            )

                                            contatore_figlie_orfane += 1

                                # ==========================================
                                # ORFANE PURE IN C
                                # ==========================================
                                idx_OC = np.where(
                                    orphans_clump_c == clump_name
                                )[0]

                                for j in idx_OC:

                                    oc_id = str(
                                        int(float(orphans_id_c[j]))
                                    )

                                    draw_ellipse(
                                        ax,
                                        wcs,
                                        cats['7MTM2TM1'],
                                        clump_name,
                                        orphans_id_c[j],
                                        'white',
                                        f"OC_{oc_id}"
                                    )

                            ax.axis('off')

                    except FileNotFoundError:

                        ax.text(
                            0.5,
                            0.5,
                            f"MANCANTE:\n{nome_fits}",
                            ha='center',
                            color='red'
                        )

                        ax.set_title(
                            f"Clump: {clump_name} | {cat_name}",
                            color='gray'
                        )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            pdf.savefig(fig)

            plt.close(fig)

    total_time = time.time() - start_time

    print(f"\nPDF Generato: {output_pdf}")
    print(f"Tempo totale: {total_time:.1f} s\n")


#Second main functions, that generate the multiplicity statistics for the clumps in the catalogues A and B, and B and C.
def multiplicity_stat_A_B(f_clump,cat_A):
    clump_stat={
        "CLUMP":[],
        "Tclump":[],
        "Dclump":[],
        "Lclump/Mclump":[],
        "RMS_map":[],
        "Surfd":[],
        "mult_stat":[]

    }
    for i,row in cat_A.iterrows():
        actual_clump=row['CLUMP']
        n = (f_clump[i,:,0]!= None).sum()-1
        mult_string =f'1->{n}'

        if actual_clump not in clump_stat["CLUMP"]:
            clump_stat["CLUMP"].append(actual_clump)
            clump_stat["Tclump"].append(row['Tclump'])
            clump_stat["Dclump"].append(row['Dclump'])
            clump_stat["Lclump/Mclump"].append(row['Lclump']/row['Mclump'])
            clump_stat["RMS_map"].append(row['RMS_map'])
            clump_stat["Surfd"].append(row['Surfd'])
            mult_stat_dict={}
            mult_stat_dict[mult_string]=1
            clump_stat["mult_stat"].append(mult_stat_dict)



        else:
            idx_actual_clump=clump_stat["CLUMP"].index(actual_clump)
            if mult_string in clump_stat["mult_stat"][idx_actual_clump]:
                clump_stat["mult_stat"][idx_actual_clump][mult_string] += 1
            else:
                clump_stat["mult_stat"][idx_actual_clump][mult_string] = 1


    return clump_stat
def multiplicity_stat_B_C(f_clump,ob_clump,cat_A):
    clump_stat={
        "CLUMP":[],
        "Tclump":[],
        "Dclump":[],
        "Lclump/Mclump":[],
        "RMS_map":[],
        "Surfd":[],
        "mult_stat":[]
    }
    for i,row in cat_A.iterrows():
        actual_clump=row['CLUMP']
        n_column = (f_clump[i,:,0]!= None).sum()
        mult_string=[]

        for j in range(n_column-1):
            n=(f_clump[i,j+1,:]!=None).sum()- 1 
            mult_string.append(f'1->{n}')

        if actual_clump not in clump_stat["CLUMP"]:
            clump_stat["CLUMP"].append(actual_clump)
            clump_stat["Tclump"].append(row['Tclump'])
            clump_stat["Dclump"].append(row['Dclump'])
            clump_stat["Lclump/Mclump"].append(row['Lclump']/row['Mclump'])
            clump_stat["RMS_map"].append(row['RMS_map'])
            clump_stat["Surfd"].append(row['Surfd'])
            mult_stat_dict={}
            for k in range(len(mult_string)):
                if mult_string[k] in mult_stat_dict:
                    mult_stat_dict[mult_string[k]] += 1
                else:
                    mult_stat_dict[mult_string[k]] = 1
            
            clump_stat["mult_stat"].append(mult_stat_dict)

        else:
            idx_actual_clump=clump_stat["CLUMP"].index(actual_clump)
            for k in range(len(mult_string)):
                if mult_string[k] in clump_stat["mult_stat"][idx_actual_clump]:
                    clump_stat["mult_stat"][idx_actual_clump][mult_string[k]] += 1
                else:
                    clump_stat["mult_stat"][idx_actual_clump][mult_string[k]] = 1
    for i in range(ob_clump.shape[0]):
        actual_clump=ob_clump[i][0]
        n=n = (ob_clump[i,:]!= None).sum()-1
        mult_string =f'1->{n}'
        if actual_clump not in clump_stat["CLUMP"]:
            clump_stat["CLUMP"].append(actual_clump)
            clump_stat["Tclump"].append(None)
            clump_stat["Dclump"].append(None)
            clump_stat["Lclump/Mclump"].append(None)
            clump_stat["RMS_map"].append(None)
            clump_stat["Surfd"].append(None)
            mult_stat_dict={}
            mult_stat_dict[mult_string] = 1
            clump_stat["mult_stat"].append(mult_stat_dict)

        else:
            idx_actual_clump=clump_stat["CLUMP"].index(actual_clump)
            if mult_string in clump_stat["mult_stat"][idx_actual_clump]:
                clump_stat["mult_stat"][idx_actual_clump][mult_string] += 1
            else:
                clump_stat["mult_stat"][idx_actual_clump][mult_string] = 1

    return clump_stat
def mult_distribution(clump_stat_A, clump_stat_B):

    N = 9

    n = np.zeros(N)
    m = np.zeros(N)
    N_sources_A = len(clump_stat_A['CLUMP'])
    N_sources_B = len(clump_stat_B['CLUMP'])

    # Conteggio per catalogo A
    for i in range(len(clump_stat_A['CLUMP'])):
        for k in range(N):
            if f'1->{k}' in clump_stat_A['mult_stat'][i]:
                n[k] += clump_stat_A['mult_stat'][i][f'1->{k}']

    # Conteggio per catalogo B
    for i in range(len(clump_stat_B['CLUMP'])):
        for k in range(N):
            if f'1->{k}' in clump_stat_B['mult_stat'][i]:
                m[k] += clump_stat_B['mult_stat'][i][f'1->{k}']

    x = np.arange(0, N, 1)

    mean_n = np.dot(n, x) / np.sum(n)
    mean_m = np.dot(m, x) / np.sum(m)

    err_mean=0
    for i in range(len(clump_stat_A['CLUMP'])):
        for k in range(N):
            if f'1->{k}' in clump_stat_A['mult_stat'][i]:
                err_mean += clump_stat_A['mult_stat'][i][f'1->{k}'] * (k - mean_n) ** 2
    err_mean_n = np.sqrt(err_mean / np.sum(n)) / np.sqrt(np.sum(n))
    err_mean=0
    for i in range(len(clump_stat_B['CLUMP'])):
        for k in range(N):
            if f'1->{k}' in clump_stat_B['mult_stat'][i]:
                err_mean += clump_stat_B['mult_stat'][i][f'1->{k}'] * (k - mean_m) ** 2
    err_mean_m = np.sqrt(err_mean / np.sum(m)) / np.sqrt(np.sum(m))

    # =========================
    # Errori poissoniani
    # =========================

    err_n = np.sqrt(n)
    err_m = np.sqrt(m)

    # Errori sulle quantità normalizzate
    err_n_norm = err_n / np.sum(n)
    err_m_norm = err_m / np.sum(m)

    # =========================
    # Plot
    # =========================

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # --- PRIMO GRAFICO (Catalogo A) ---

    ax[0].bar(
        x,
        n / np.sum(n),
        yerr=err_n_norm,
        capsize=5,
        color='skyblue',
        edgecolor='black',
        width=0.8
    )

    ax[0].axvline(
        mean_n,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean multiplicity A->B: {mean_n:.2f} ± {err_mean_n:.2f}'
    )

    ax[0].set_xticks(x)
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('Normalized entries')
    ax[0].set_title('Catalog A -> B')
    ax[0].legend()

    # --- SECONDO GRAFICO (Catalogo B) ---

    ax[1].bar(
        x,
        m / np.sum(m),
        yerr=err_m_norm,
        capsize=5,
        color='coral',
        edgecolor='black',
        width=0.8
    )

    ax[1].axvline(
        mean_m,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean multiplicity B->C: {mean_m:.2f} ± {err_mean_m:.2f}'
    )

    ax[1].set_xticks(x)
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('Normalized entries')
    ax[1].set_title('Catalog B -> C')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    print(mean_n, mean_m)

    return


#Third main functions, giving the final results of the statistics created
def plot_multiplicity_vs_properties(
    clump_stat_AB,
    clump_stat_BC,
    multiplicity='1->2'
):

    import numpy as np
    import matplotlib.pyplot as plt

    properties_keys = [
        'Lclump/Mclump',
        'Tclump',
        'Dclump',
        'RMS_map',
        'Surfd'
    ]

    property_labels = [
        'L/M',
        'Tclump',
        'Dclump',
        'RMS_map',
        'Surfd'
    ]

    n_bins_AB = max(
        3,
        int(1 + np.log2(len(clump_stat_AB["CLUMP"])))
    )

    n_bins_BC = max(
        3,
        int(1 + np.log2(len(clump_stat_BC["CLUMP"])))
    )

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(18, 16)
    )

    for row_idx, (prop_key, prop_label) in enumerate(
        zip(properties_keys, property_labels)
    ):

        # ======================================================
        # VALID VALUES FOR BINNING
        # ======================================================

        def select_valid_values(clump_stat):

            valid = []

            for value in clump_stat[prop_key]:

                try:
                    value = float(value)
                except (
                    ValueError,
                    TypeError
                ):
                    continue

                if not np.isfinite(value):
                    continue

                if value == -999:
                    continue

                if prop_key == 'Lclump/Mclump':
                    if value >= 1:
                        continue

                elif prop_key == 'RMS_map':
                    if value >= 5:
                        continue

                elif prop_key == 'Dclump':
                    if not (0.1 < value < 0.6):
                        continue

                elif prop_key == 'Surfd':
                    if value >= 4:
                        continue

                valid.append(value)

            return np.array(valid)

        all_values_AB = select_valid_values(
            clump_stat_AB
        )

        all_values_BC = select_valid_values(
            clump_stat_BC
        )

        if (
            len(all_values_AB) == 0
            or
            len(all_values_BC) == 0
        ):
            print(
                f"Skipping {prop_key}: "
                f"no valid values."
            )
            continue

        bins_AB = np.linspace(
            np.min(all_values_AB),
            np.max(all_values_AB),
            n_bins_AB + 1
        )

        bins_BC = np.linspace(
            np.min(all_values_BC),
            np.max(all_values_BC),
            n_bins_BC + 1
        )

        bin_centers_AB = (
            bins_AB[:-1] +
            bins_AB[1:]
        ) / 2

        bin_centers_BC = (
            bins_BC[:-1] +
            bins_BC[1:]
        ) / 2

        # ======================================================
        # PROCESS CATALOG
        # ======================================================

        def process_catalog(
            clump_stat,
            bins,
            n_bins
        ):

            multiplicity_sum_per_bin = np.zeros(
                n_bins,
                dtype=float
            )

            source_sum_per_bin = np.zeros(
                n_bins,
                dtype=float
            )

            counts = np.zeros(
                n_bins,
                dtype=int
            )

            for i in range(
                len(clump_stat["CLUMP"])
            ):

                try:
                    prop_val = float(
                        clump_stat[prop_key][i]
                    )
                except (
                    ValueError,
                    TypeError
                ):
                    continue

                if not np.isfinite(prop_val):
                    continue

                if prop_val == -999:
                    continue

                if prop_key == 'Lclump/Mclump':

                    if prop_val >= 1:
                        continue

                elif prop_key == 'RMS_map':

                    if prop_val >= 5:
                        continue

                elif prop_key == 'Dclump':

                    if not (
                        0.1 < prop_val < 0.6
                    ):
                        continue

                elif prop_key == 'Surfd':

                    if prop_val >= 4:
                        continue

                mult_dict = (
                    clump_stat["mult_stat"][i]
                )

                bin_idx = (
                    np.digitize(
                        prop_val,
                        bins
                    ) - 1
                )

                if bin_idx == n_bins:
                    bin_idx = n_bins - 1

                if not (
                    0 <= bin_idx < n_bins
                ):
                    continue

                counts[bin_idx] += 1

                if not isinstance(
                    mult_dict,
                    dict
                ):
                    continue

                selected_mult = (
                    mult_dict.get(
                        multiplicity,
                        0
                    )
                )

                total_sources = 0

                for key, value in mult_dict.items():

                    if (
                        isinstance(key, str)
                        and
                        key.startswith('1->')
                    ):

                        try:
                            total_sources += float(value)
                        except (
                            ValueError,
                            TypeError
                        ):
                            pass

                multiplicity_sum_per_bin[
                    bin_idx
                ] += selected_mult

                source_sum_per_bin[
                    bin_idx
                ] += total_sources

            fractions = np.full(
                n_bins,
                np.nan
            )

            errors = np.zeros(
                n_bins
            )

            for k in range(n_bins):

                if (
                    source_sum_per_bin[k]
                    > 0
                ):

                    fractions[k] = (
                        multiplicity_sum_per_bin[k]
                        /
                        source_sum_per_bin[k]
                    )

            return (
                fractions,
                errors,
                counts
            )

        # ======================================================
        # AB
        # ======================================================

        frac_AB, err_AB, count_AB = (
            process_catalog(
                clump_stat_AB,
                bins_AB,
                n_bins_AB
            )
        )

        ax1 = axes[row_idx, 0]

        ax1.bar(
            bin_centers_AB,
            frac_AB,
            yerr=err_AB,
            capsize=4,
            width=np.diff(
                bins_AB
            ).mean() * 0.8,
            color='steelblue',
            edgecolor='black',
            alpha=0.7
        )

        ax1.set_xlabel(
            prop_label
        )

        ax1.set_ylabel(
            f'Fraction of {multiplicity}'
        )

        ax1.set_title(
            f'A → B Fraction of {multiplicity} vs {prop_label}'
        )

        ax1.grid(
            True,
            alpha=0.3
        )

        ax1_twin = ax1.twinx()

        ax1_twin.plot(
            bin_centers_AB,
            count_AB,
            'r-o'
        )

        ax1_twin.set_ylabel(
            'Number of clumps',
            color='red'
        )

        ax1_twin.tick_params(
            axis='y',
            labelcolor='red'
        )

        # ======================================================
        # BC
        # ======================================================

        frac_BC, err_BC, count_BC = (
            process_catalog(
                clump_stat_BC,
                bins_BC,
                n_bins_BC
            )
        )

        ax2 = axes[row_idx, 1]

        ax2.bar(
            bin_centers_BC,
            frac_BC,
            yerr=err_BC,
            capsize=4,
            width=np.diff(
                bins_BC
            ).mean() * 0.8,
            color='coral',
            edgecolor='black',
            alpha=0.7
        )

        ax2.set_xlabel(
            prop_label
        )

        ax2.set_ylabel(
            f'Fraction of {multiplicity}'
        )

        ax2.set_title(
            f'B → C Fraction of {multiplicity} vs {prop_label}'
        )

        ax2.grid(
            True,
            alpha=0.3
        )

        ax2_twin = ax2.twinx()

        ax2_twin.plot(
            bin_centers_BC,
            count_BC,
            'r-o'
        )

        ax2_twin.set_ylabel(
            'Number of clumps',
            color='red'
        )

        ax2_twin.tick_params(
            axis='y',
            labelcolor='red'
        )

    plt.tight_layout()
    plt.show()

    print()

    print(
        f"Total AB clumps: "
        f"{len(clump_stat_AB['CLUMP'])}"
    )

    print(
        f"Total BC clumps: "
        f"{len(clump_stat_BC['CLUMP'])}"
    )

    print()

    print(
        f"Multiplicity selected: "
        f"{multiplicity}"
    )
def plot_multiplicity_vs_properties_log_with_fit(
    clump_stat_AB,
    clump_stat_BC,
    multiplicity='1->2-4',
    run_fit=True
):

    import numpy as np
    import matplotlib.pyplot as plt
    import re
    from scipy.optimize import curve_fit
    from scipy.stats import chi2

    properties_keys = [
        'Lclump/Mclump',
        'Tclump',
        'Dclump',
        'RMS_map',
        'Surfd'
    ]

    property_labels = [
    r'L/M [$L_\odot/M_\odot$]',
    r'Tclump [K]',
    r'Dclump [pc]',
    r'RMS$_{map}$ [MJy/Sr]',
    r'Surfd [g/cm$^2$]'
    ]   

    log_properties = ['Lclump/Mclump', 'Dclump', 'Surfd']

    def linear_model(x, m, q):
        return m * x + q

    range_match = re.match(r'^1->(\d+)-(\d+)$', multiplicity)
    plus_match = re.match(r'^1->(\d+)\+$', multiplicity)

    if multiplicity in ['1->0', 'core_disappearance']:
        mult_label = 'Core disappearance (1->0)'
    elif multiplicity in ['1->1', 'no_fragmentation']:
        mult_label = 'No fragmentation (1->1)'
    elif multiplicity in ['1->2-4', 'moderate', 'moderate_fragmentation']:
        mult_label = 'Moderate fragmentation (1->2-4)'
    elif multiplicity in ['1->5+', 'high', 'high_fragmentation']:
        mult_label = 'High fragmentation (1->5+)'
    elif range_match:
        mult_label = f'Fraction of {multiplicity}'
    elif plus_match:
        mult_label = f'Fraction of {multiplicity}'
    else:
        mult_label = f'Fraction of {multiplicity}'

    n_bins_AB = max(3, int(1 + np.log2(len(clump_stat_AB["CLUMP"]))))
    n_bins_BC = max(3, int(1 + np.log2(len(clump_stat_BC["CLUMP"]))))

    fig, axes = plt.subplots(5, 2, figsize=(18, 24))

    for row_idx, (prop_key, prop_label) in enumerate(zip(properties_keys, property_labels)):

        def select_valid_values(clump_stat):
            valid = []
            for value in clump_stat[prop_key]:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(value) or value == -999:
                    continue
                if prop_key in log_properties:
                    if value <= 0:
                        continue
                else:
                    if prop_key == 'RMS_map' and value >= 5:
                        continue
                valid.append(value)
            return np.array(valid)

        all_values_AB = select_valid_values(clump_stat_AB)
        all_values_BC = select_valid_values(clump_stat_BC)

        if len(all_values_AB) == 0 or len(all_values_BC) == 0:
            print(f"Skipping {prop_key}: no valid values.")
            continue

        if prop_key in log_properties:
            bins_AB = np.logspace(np.log10(np.min(all_values_AB)), np.log10(np.max(all_values_AB)), n_bins_AB + 1)
            log_bins_AB = np.log10(bins_AB)
            log_centers_AB = (log_bins_AB[:-1] + log_bins_AB[1:]) / 2
            log_widths_AB = np.diff(log_bins_AB)
            bin_centers_AB = (10**(log_centers_AB - 0.4 * log_widths_AB) + 10**(log_centers_AB + 0.4 * log_widths_AB)) / 2
            widths_AB = 10**(log_centers_AB + 0.4 * log_widths_AB) - 10**(log_centers_AB - 0.4 * log_widths_AB)

            bins_BC = np.logspace(np.log10(np.min(all_values_BC)), np.log10(np.max(all_values_BC)), n_bins_BC + 1)
            log_bins_BC = np.log10(bins_BC)
            log_centers_BC = (log_bins_BC[:-1] + log_bins_BC[1:]) / 2
            log_widths_BC = np.diff(log_bins_BC)
            bin_centers_BC = (10**(log_centers_BC - 0.4 * log_widths_BC) + 10**(log_centers_BC + 0.4 * log_widths_BC)) / 2
            widths_BC = 10**(log_centers_BC + 0.4 * log_widths_BC) - 10**(log_centers_BC - 0.4 * log_widths_BC)
        else:
            bins_AB = np.linspace(np.min(all_values_AB), np.max(all_values_AB), n_bins_AB + 1)
            bin_centers_AB = (bins_AB[:-1] + bins_AB[1:]) / 2
            widths_AB = np.diff(bins_AB) * 0.8

            bins_BC = np.linspace(np.min(all_values_BC), np.max(all_values_BC), n_bins_BC + 1)
            bin_centers_BC = (bins_BC[:-1] + bins_BC[1:]) / 2
            widths_BC = np.diff(bins_BC) * 0.8

        def process_catalog(clump_stat, bins, n_bins):
            multiplicity_sum_per_bin = np.zeros(n_bins, dtype=float)
            source_sum_per_bin = np.zeros(n_bins, dtype=float)
            counts = np.zeros(n_bins, dtype=int)

            for i in range(len(clump_stat["CLUMP"])):
                try:
                    prop_val = float(clump_stat[prop_key][i])
                except (ValueError, TypeError):
                    continue
                if not np.isfinite(prop_val) or prop_val == -999:
                    continue
                if prop_key in log_properties and prop_val <= 0:
                    continue
                if prop_key == 'RMS_map' and prop_val >= 5:
                    continue

                mult_dict = clump_stat["mult_stat"][i]
                bin_idx = np.digitize(prop_val, bins) - 1
                if bin_idx == n_bins:
                    bin_idx = n_bins - 1
                if not (0 <= bin_idx < n_bins):
                    continue

                counts[bin_idx] += 1
                if not isinstance(mult_dict, dict):
                    continue

                selected_mult = 0
                total_sources = 0

                for key, value in mult_dict.items():
                    if not isinstance(key, str) or not key.startswith('1->'):
                        continue
                    try:
                        val_float = float(value)
                    except (ValueError, TypeError):
                        continue

                    total_sources += val_float

                    try:
                        n_dest = int(key.split('->')[1])
                    except (ValueError, IndexError):
                        continue

                    is_match = False
                    if multiplicity in ['1->0', 'core_disappearance']:
                        is_match = (n_dest == 0)
                    elif multiplicity in ['1->1', 'no_fragmentation']:
                        is_match = (n_dest == 1)
                    elif multiplicity in ['1->2-4', 'moderate', 'moderate_fragmentation']:
                        is_match = (2 <= n_dest <= 4)
                    elif multiplicity in ['1->5+', 'high', 'high_fragmentation']:
                        is_match = (n_dest >= 5)
                    else:
                        r_match = re.match(r'^1->(\d+)-(\d+)$', multiplicity)
                        p_match = re.match(r'^1->(\d+)\+$', multiplicity)
                        if r_match:
                            is_match = (int(r_match.group(1)) <= n_dest <= int(r_match.group(2)))
                        elif p_match:
                            is_match = (n_dest >= int(p_match.group(1)))
                        else:
                            is_match = (key == multiplicity)

                    if is_match:
                        selected_mult += val_float

                multiplicity_sum_per_bin[bin_idx] += selected_mult
                source_sum_per_bin[bin_idx] += total_sources

            fractions = np.full(n_bins, np.nan)
            errors = np.zeros(n_bins)

            for k in range(n_bins):
                n_parent = source_sum_per_bin[k]
                if n_parent > 0:
                    fractions[k] = multiplicity_sum_per_bin[k] / n_parent
                    if n_parent > 1:
                        p = fractions[k]
                        errors[k] = np.sqrt((p * (1 - p)) / (n_parent - 1))
                    else:
                        errors[k] = 0.0

            return fractions, errors, counts

        # ======================================================
        # AB (PLOTTING AND FITTING)
        # ======================================================
        frac_AB, err_AB, count_AB = process_catalog(clump_stat_AB, bins_AB, n_bins_AB)
        ax1 = axes[row_idx, 0]

        ax1.bar(
            bin_centers_AB,
            frac_AB,
            yerr=err_AB,
            capsize=4,
            width=widths_AB,
            color='steelblue',
            edgecolor='black',
            alpha=0.7,
            label='Data'
        )

        if prop_key in log_properties:
            ax1.set_xscale('log')

        if run_fit:
            valid_mask_AB = np.isfinite(frac_AB) & np.isfinite(err_AB) & (err_AB > 0)
            if np.sum(valid_mask_AB) >= 2:
                is_log = prop_key in log_properties
                x_fit = np.log10(bin_centers_AB[valid_mask_AB]) if is_log else bin_centers_AB[valid_mask_AB]
                y_fit = frac_AB[valid_mask_AB]
                weights = err_AB[valid_mask_AB]

                try:
                    popt, pcov = curve_fit(linear_model, x_fit, y_fit, sigma=weights, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))
                    
                    if is_log:
                        x_line = np.logspace(np.log10(np.min(bin_centers_AB[valid_mask_AB])), np.log10(np.max(bin_centers_AB[valid_mask_AB])), 100)
                        y_line = linear_model(np.log10(x_line), *popt)
                        space_str = "Spazio Log10"
                    else:
                        x_line = np.linspace(np.min(bin_centers_AB[valid_mask_AB]), np.max(bin_centers_AB[valid_mask_AB]), 100)
                        y_line = linear_model(x_line, *popt)
                        space_str = "Spazio Lineare"
                    
                    fit_label = (f"Fit Lineare ({space_str})\n"
                                 f"m = {popt[0]:.2e} ± {perr[0]:.2e}\n"
                                 f"q = {popt[1]:.2e} ± {perr[1]:.2e}")
                    
                    ax1.plot(x_line, y_line, color='darkblue', linestyle='--', linewidth=2, label=fit_label)

                    y_model = linear_model(x_fit, *popt)
                    chi2_obs = np.sum(((y_fit - y_model) / weights) ** 2)
                    dof = len(x_fit) - 2

                    if dof > 0:
                        chi2_crit = chi2.ppf(0.95, dof)
                        status = "PASS" if chi2_obs <= chi2_crit else "FAIL"
                        box_text = f"Chi2: {chi2_obs:.2f}\nCrit (0.05): {chi2_crit:.2f}\nStatus: {status}"
                    else:
                        box_text = f"Chi2: {chi2_obs:.2f}\ndof: {dof} (No Crit)"
                    
                    ax1.text(0.95, 0.95, box_text, transform=ax1.transAxes, verticalalignment='top', 
                             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
                except Exception:
                    pass

        ax1.set_xlabel(prop_label)
        ax1.set_ylabel(f'Fraction: {mult_label}')
        ax1.set_title(f'A → B | {mult_label} vs {prop_label}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)

        # ======================================================
        # BC (PLOTTING AND FITTING)
        # ======================================================
        frac_BC, err_BC, count_BC = process_catalog(clump_stat_BC, bins_BC, n_bins_BC)
        ax2 = axes[row_idx, 1]

        ax2.bar(
            bin_centers_BC,
            frac_BC,
            yerr=err_BC,
            capsize=4,
            width=widths_BC,
            color='coral',
            edgecolor='black',
            alpha=0.7,
            label='Data'
        )

        if prop_key in log_properties:
            ax2.set_xscale('log')

        if run_fit:
            valid_mask_BC = np.isfinite(frac_BC) & np.isfinite(err_BC) & (err_BC > 0)
            if np.sum(valid_mask_BC) >= 2:
                is_log = prop_key in log_properties
                x_fit = np.log10(bin_centers_BC[valid_mask_BC]) if is_log else bin_centers_BC[valid_mask_BC]
                y_fit = frac_BC[valid_mask_BC]
                weights = err_BC[valid_mask_BC]

                try:
                    popt, pcov = curve_fit(linear_model, x_fit, y_fit, sigma=weights, absolute_sigma=True)
                    perr = np.sqrt(np.diag(pcov))
                    
                    if is_log:
                        x_line = np.logspace(np.log10(np.min(bin_centers_BC[valid_mask_BC])), np.log10(np.max(bin_centers_BC[valid_mask_BC])), 100)
                        y_line = linear_model(np.log10(x_line), *popt)
                        space_str = "Spazio Log10"
                    else:
                        x_line = np.linspace(np.min(bin_centers_BC[valid_mask_BC]), np.max(bin_centers_BC[valid_mask_BC]), 100)
                        y_line = linear_model(x_line, *popt)
                        space_str = "Spazio Lineare"
                    
                    fit_label = (f"Fit Lineare ({space_str})\n"
                                 f"m = {popt[0]:.2e} ± {perr[0]:.2e}\n"
                                 f"q = {popt[1]:.2e} ± {perr[1]:.2e}")
                    
                    ax2.plot(x_line, y_line, color='darkred', linestyle='--', linewidth=2, label=fit_label)

                    y_model = linear_model(x_fit, *popt)
                    chi2_obs = np.sum(((y_fit - y_model) / weights) ** 2)
                    dof = len(x_fit) - 2

                    if dof > 0:
                        chi2_crit = chi2.ppf(0.95, dof)
                        status = "PASS" if chi2_obs <= chi2_crit else "FAIL"
                        box_text = f"Chi2: {chi2_obs:.2f}\nCrit (0.05): {chi2_crit:.2f}\nStatus: {status}"
                    else:
                        box_text = f"Chi2: {chi2_obs:.2f}\ndof: {dof} (No Crit)"
                    
                    ax2.text(0.95, 0.95, box_text, transform=ax2.transAxes, verticalalignment='top', 
                             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
                except Exception:
                    pass

        ax2.set_xlabel(prop_label)
        ax2.set_ylabel(f'Fraction: {mult_label}')
        ax2.set_title(f'B → C | {mult_label} vs {prop_label}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()

    print()
    print(f"Total AB clumps: {len(clump_stat_AB['CLUMP'])}")
    print(f"Total BC clumps: {len(clump_stat_BC['CLUMP'])}")
    print()
    print(f"Multiplicity selected: {multiplicity} ({mult_label})")
def joint_pdf_mult_prop(clump_stat_AB, 
                                clump_stat_BC):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # ==========================================================
    # Total number of sources in each catalogue
    # ==========================================================
    N_AB = 0

    for mult_dict in clump_stat_AB["mult_stat"]:

        for value in mult_dict.values():

            N_AB += value

    N_BC = 0

    for mult_dict in clump_stat_BC["mult_stat"]:

        for value in mult_dict.values():

            N_BC += value

    # ==========================================================
    # Properties
    # ==========================================================
    properties = [
        "Tclump",
        "Dclump",
        "Lclump/Mclump",
        "RMS_map",
        "Surfd"
    ]

    property_labels = [
        "Tclump",
        "Dclump",
        "L/M",
        "RMS_map",
        "Surface density"
    ]

    # ==========================================================
    # Figure setup
    # ==========================================================
    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(18, 28),
        sharex=False
    )

    datasets = [
        (clump_stat_AB, "A → B", N_AB),
        (clump_stat_BC, "B → C", N_BC)
    ]

    # ==========================================================
    # Main loop
    # ==========================================================
    for col, (clump_stat, title, Ntot) in enumerate(datasets):

        for row, (prop, label) in enumerate(
            zip(properties, property_labels)
        ):

            ax = axes[row, col]

            x_vals = []

            y_vals = []

            weights = []

            # ==================================================
            # Build arrays
            # ==================================================
            for i in range(len(clump_stat["CLUMP"])):

                prop_value = clump_stat[prop][i]

                # ----------------------------------------------
                # Remove invalid values
                # ----------------------------------------------
                if prop_value is None:
                    continue

                if prop == "Tclump" and prop_value == -999.0:
                    continue

                if prop == "Dclump" and prop_value == -999.0:
                    continue

                # ----------------------------------------------
                # Same cuts as orphan function
                # ----------------------------------------------
                if prop == "Lclump/Mclump":

                    if prop_value >= 1:
                        continue

                elif prop == "RMS_map":

                    if prop_value >= 5:
                        continue

                elif prop == "Dclump":

                    if not (0.1 < prop_value < 0.6):
                        continue

                elif prop == "Surfd":

                    if prop_value >= 4:
                        continue

                mult_dict = clump_stat["mult_stat"][i]

                for key, count in mult_dict.items():

                    try:

                        n = int(key.split("->")[1])

                    except:

                        continue

                    # ==========================================
                    # AXES SWAPPED
                    # ==========================================
                    x_vals.append(prop_value)

                    y_vals.append(n)

                    # ==========================================
                    # NORMALIZATION
                    # ==========================================
                    weights.append(count / Ntot)

            # ==================================================
            # Convert to arrays
            # ==================================================
            x_vals = np.array(x_vals)

            y_vals = np.array(y_vals)

            weights = np.array(weights)

            # ==================================================
            # Skip empty panels
            # ==================================================
            if len(x_vals) == 0:

                ax.set_visible(False)

                continue

            # ==================================================
            # Define bins
            # ==================================================
            x_min = np.min(x_vals)

            x_max = np.max(x_vals)

            x_bins = np.linspace(
                x_min,
                x_max,
                50
            )

            y_bins = np.arange(
                0,
                np.max(y_vals) + 2
            ) - 0.5

            # ==================================================
            # Heatmap
            # ==================================================
            h = ax.hist2d(
                x_vals,
                y_vals,
                bins=[x_bins, y_bins],
                weights=weights,
                cmap="viridis",#inferno,magma,plasma,cividis
                norm=LogNorm(),
                cmin=1e-10
            )

            # ==================================================
            # Colorbar
            # ==================================================
            cbar = fig.colorbar(
                h[3],
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.set_label(
                "Normalized number of sources",
                fontsize=10
            )

            # ==================================================
            # Labels
            # ==================================================
            ax.set_xlabel(
                label,
                fontsize=11
            )

            ax.set_ylabel(
                "Multiplicity n in 1→n",
                fontsize=11
            )

            if row == 0:

                ax.set_title(
                    title,
                    fontsize=16
                )

            # ==================================================
            # Axis limits
            # ==================================================
            if prop == "Lclump/Mclump":

                ax.set_xlim(0, 1)

            elif prop == "RMS_map":

                ax.set_xlim(0, 5)

            elif prop == "Dclump":

                ax.set_xlim(0.1, 0.6)

            elif prop == "Surfd":

                ax.set_xlim(0, 4)

            # ==================================================
            # Improve appearance
            # ==================================================
            ax.set_ylim(
                -0.5,
                np.max(y_vals) + 0.5
            )

            ax.set_yticks(
                np.arange(0, np.max(y_vals) + 1, 1)
            )

            ax.tick_params(
                axis='both',
                labelsize=10
            )

            ax.grid(
                alpha=0.25
            )

            ax.set_facecolor("black")

    # ==========================================================
    # Final layout
    # ==========================================================
    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.25
    )

    plt.show()
def conditional_pdf_mult_prop(
    clump_stat_AB,
    clump_stat_BC
):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    properties = [
        "Tclump",
        "Dclump",
        "Lclump/Mclump",
        "RMS_map",
        "Surfd"
    ]

    property_labels = [
        "Tclump",
        "Dclump",
        "L/M",
        "RMS_map",
        "Surface density"
    ]

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(18, 28),
        sharex=False
    )

    datasets = [
        (clump_stat_AB, "A → B"),
        (clump_stat_BC, "B → C")
    ]

    for col, (clump_stat, title) in enumerate(
        datasets
    ):

        for row, (prop, label) in enumerate(
            zip(
                properties,
                property_labels
            )
        ):

            ax = axes[row, col]

            x_vals = []
            y_vals = []
            counts = []

            # ==========================================
            # BUILD ARRAYS
            # ==========================================

            for i in range(
                len(clump_stat["CLUMP"])
            ):

                prop_value = (
                    clump_stat[prop][i]
                )

                try:

                    prop_value = float(
                        prop_value
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                if not np.isfinite(
                    prop_value
                ):
                    continue

                if prop_value == -999:
                    continue

                if prop == "Lclump/Mclump":

                    if prop_value >= 1:
                        continue

                elif prop == "RMS_map":

                    if prop_value >= 5:
                        continue

                elif prop == "Dclump":

                    if not (
                        0.1
                        <
                        prop_value
                        <
                        0.6
                    ):
                        continue

                elif prop == "Surfd":

                    if prop_value >= 4:
                        continue

                mult_dict = (
                    clump_stat["mult_stat"][i]
                )

                if not isinstance(
                    mult_dict,
                    dict
                ):
                    continue

                for key, count in (
                    mult_dict.items()
                ):

                    try:

                        n = int(
                            key.split(
                                "->"
                            )[1]
                        )

                    except:

                        continue

                    x_vals.append(
                        prop_value
                    )

                    y_vals.append(
                        n
                    )

                    counts.append(
                        count
                    )

            x_vals = np.array(
                x_vals
            )

            y_vals = np.array(
                y_vals
            )

            counts = np.array(
                counts
            )

            if len(x_vals) == 0:

                ax.set_visible(
                    False
                )

                continue

            # ==========================================
            # BINS
            # ==========================================

            x_min = np.min(
                x_vals
            )

            x_max = np.max(
                x_vals
            )

            x_bins = np.linspace(
                x_min,
                x_max,
                50
            )

            y_max = int(
                np.max(
                    y_vals
                )
            )

            y_bins = np.arange(
                0,
                y_max + 2
            ) - 0.5

            nx = len(
                x_bins
            ) - 1

            ny = len(
                y_bins
            ) - 1

            H = np.zeros(
                (
                    ny,
                    nx
                ),
                dtype=float
            )

            # ==========================================
            # FILL MATRIX
            # ==========================================

            x_idx = np.digitize(
                x_vals,
                x_bins
            ) - 1

            y_idx = np.digitize(
                y_vals,
                y_bins
            ) - 1

            valid = (
                (x_idx >= 0)
                &
                (x_idx < nx)
                &
                (y_idx >= 0)
                &
                (y_idx < ny)
            )

            x_idx = x_idx[
                valid
            ]

            y_idx = y_idx[
                valid
            ]

            counts = counts[
                valid
            ]

            for ix, iy, c in zip(
                x_idx,
                y_idx,
                counts
            ):

                H[
                    iy,
                    ix
                ] += c

            # ==========================================
            # COLUMN NORMALIZATION
            #
            # P(n | property)
            # ==========================================

            column_sum = np.sum(
                H,
                axis=0
            )

            nonzero = (
                column_sum > 0
            )

            H[:, nonzero] /= (
                column_sum[
                    nonzero
                ]
            )

            H[
                H == 0
            ] = np.nan

            # ==========================================
            # PLOT
            # ==========================================

            im = ax.pcolormesh(
                x_bins,
                y_bins,
                H,
                cmap="viridis",
                norm=LogNorm(
                    vmin=np.nanmin(
                        H
                    ),
                    vmax=np.nanmax(
                        H
                    )
                ),
                shading="auto"
            )

            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )

            cbar.set_label(
                "P(n | property)",
                fontsize=10
            )

            ax.set_xlabel(
                label,
                fontsize=11
            )

            ax.set_ylabel(
                "Multiplicity n in 1→n",
                fontsize=11
            )

            if row == 0:

                ax.set_title(
                    title,
                    fontsize=16
                )

            if prop == "Lclump/Mclump":

                ax.set_xlim(
                    0,
                    1
                )

            elif prop == "RMS_map":

                ax.set_xlim(
                    0,
                    5
                )

            elif prop == "Dclump":

                ax.set_xlim(
                    0.1,
                    0.6
                )

            elif prop == "Surfd":

                ax.set_xlim(
                    0,
                    4
                )

            ax.set_ylim(
                -0.5,
                y_max + 0.5
            )

            ax.set_yticks(
                np.arange(
                    0,
                    y_max + 1,
                    1
                )
            )

            ax.tick_params(
                axis='both',
                labelsize=10
            )

            ax.grid(
                alpha=0.25
            )

            ax.set_facecolor(
                "black"
            )

    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.25
    )

    plt.show()


#Additional Functions. Not necessary, however may be useful. The first one is used to find the best combination of n and m values that minimize the sum of certain counts. The others are related to the overlap between the sources in different catalogs.
def best_m_and_n(start,stop,step):
    n_values = np.arange(start, stop, step)
    m_values = np.arange(start, stop, step)

    sum_values = 0
    best_sum = np.inf
    best_combo = None

    n_summary = {}
    m_summary = {}

    for n_val in n_values:
        for m_val in m_values:
            f_clump_tmp, f_id_tmp, ob_clump_tmp, ob_id_tmp, oc_clump_tmp, oc_id_tmp = operation_a_pro(
                n_val, m_val, 20, 20
            )

            n_B = int(np.count_nonzero((ob_clump_tmp[:, 0] != None) & (ob_clump_tmp[:, 0] != 'None')))
            n_C = int(np.count_nonzero((oc_clump_tmp != None) & (oc_clump_tmp != 'None')))

            b_parents = {}
            for i in range(f_clump_tmp.shape[0]):
               for j in range(1, f_clump_tmp.shape[1]):
                    clump_b = f_clump_tmp[i, j, 0]
                    id_b = f_id_tmp[i, j, 0]
                    if clump_b is None or clump_b == 'None' or id_b is None or id_b == 'None':
                        continue
                    key = (clump_b, id_b)
                    b_parents.setdefault(key, set()).add(i)
            n_BMF = sum(1 for parents in b_parents.values() if len(parents) > 1)

            c_parents = {}
            for i in range(f_clump_tmp.shape[0]):
                for j in range(1, f_clump_tmp.shape[1]):
                    clump_b = f_clump_tmp[i, j, 0]
                    id_b = f_id_tmp[i, j, 0]
                    if clump_b is None or clump_b == 'None' or id_b is None or id_b == 'None':
                        continue
                    parent_b = (clump_b, id_b)

                    for k in range(1, f_clump_tmp.shape[2]):
                        clump_c = f_clump_tmp[i, j, k]
                        id_c = f_id_tmp[i, j, k]
                        if clump_c is None or clump_c == 'None' or id_c is None or id_c == 'None':
                            continue
                        c_parents.setdefault((clump_c, id_c), set()).add(parent_b)

            for r in range(ob_clump_tmp.shape[0]):
                clump_b = ob_clump_tmp[r, 0]
                id_b = ob_id_tmp[r, 0]
                if clump_b is None or clump_b == 'None' or id_b is None or id_b == 'None':
                    continue
                parent_b = (clump_b, id_b)

                for k in range(1, ob_clump_tmp.shape[1]):
                    clump_c = ob_clump_tmp[r, k]
                    id_c = ob_id_tmp[r, k]
                    if clump_c is None or clump_c == 'None' or id_c is None or id_c == 'None':
                        continue
                    c_parents.setdefault((clump_c, id_c), set()).add(parent_b)

            n_CMF = sum(1 for parents in c_parents.values() if len(parents) > 1)

            row_sum = n_B + n_C + n_BMF + n_CMF
            sum_values += row_sum

            if row_sum < best_sum:
                best_sum = row_sum
                best_combo = (n_val, m_val)

            if n_val not in n_summary:
                n_summary[n_val] = (n_B, n_BMF)
            elif n_summary[n_val] != (n_B, n_BMF):
                print(f"Warning: inconsistent B counts for n={n_val:.1f} across m values")

            if m_val not in m_summary:
                m_summary[m_val] = (n_C, n_CMF)
            elif m_summary[m_val] != (n_C, n_CMF):
                print(f"Warning: inconsistent C counts for m={m_val:.1f} across n values")

            print(
                f"n={n_val:.1f}, m={m_val:.1f} -> "
                f"n_B={n_B}, n_BMF={n_BMF}, n_C={n_C}, n_CMF={n_CMF}, sum={row_sum}"
            )

    n_keys = sorted(n_summary)
    m_keys = sorted(m_summary)

    n_B_vals = [n_summary[n][0] for n in n_keys]
    n_BMF_vals = [n_summary[n][1] for n in n_keys]

    n_C_vals = [m_summary[m][0] for m in m_keys]
    n_CMF_vals = [m_summary[m][1] for m in m_keys]
    n_arr = np.array(n_values)
    m_arr = np.array(m_values)

    n_arr = np.array(n_values)
    m_arr = np.array(m_values)

    n_B = np.array(n_B_vals)
    n_BMF = np.array(n_BMF_vals)
    n_C = np.array(n_C_vals)
    n_CMF = np.array(n_CMF_vals)

    n_B_mat = np.tile(n_B[:, None], (1, m_arr.size))
    n_BMF_mat = np.tile(n_BMF[:, None], (1, m_arr.size))
    n_C_mat = np.tile(n_C[None, :], (n_arr.size, 1))
    n_CMF_mat = np.tile(n_CMF[None, :], (n_arr.size, 1))
    sum_mat = n_B_mat + n_BMF_mat + n_C_mat + n_CMF_mat

    titles = [
    "n_B",
    "n_BMF",
    "n_C",
    "n_CMF",
    "n_B + n_BMF + n_C + n_CMF"
]
    matrices = [n_B_mat, n_BMF_mat, n_C_mat, n_CMF_mat, sum_mat]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax_i, matrix, title in zip(axes, matrices, titles):
        im = ax_i.imshow(
        matrix,
        origin='lower',
        aspect='auto',
        cmap='viridis',
        extent=[m_arr[0], m_arr[-1], n_arr[0], n_arr[-1]]
    )
        ax_i.set_title(title)
        ax_i.set_xlabel('m')
        ax_i.set_ylabel('n')
        ax_i.set_xticks(m_arr)
        ax_i.set_yticks(n_arr)
        fig.colorbar(im, ax=ax_i, label='count')

    if len(axes) > len(matrices):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

    sum_mean_vs_n = sum_mat.mean(axis=1)
    sum_mean_vs_m = sum_mat.mean(axis=0)

    fig, (ax_n, ax_m) = plt.subplots(1, 2, figsize=(14, 5))

    ax_n.plot(n_arr, n_B, '-o', label='n_orphans_B')
    ax_n.plot(n_arr, n_BMF, '-o', label='n_B_with_multi_fathers')
    ax_n.plot(n_arr, sum_mean_vs_n, '-o', label='avg total sum (all 4 components)')
    ax_n.set_xlabel('n')
    ax_n.set_ylabel('count')
    #ax_n.set_yscale('log')
    ax_n.set_title('Counts vs n')
    ax_n.legend()
    ax_n.grid(True)

    ax_m.plot(m_arr, n_C, '-o', label='n_orphans_C')
    ax_m.plot(m_arr, n_CMF, '-o', label='n_C_with_multi_fathers')
    ax_m.plot(m_arr, sum_mean_vs_m, '-o', label='avg total sum (all 4 components)')
    ax_m.set_xlabel('m')
    ax_m.set_ylabel('count')
    ax_m.set_title('Counts vs m')
    ax_m.set_yscale('log')
    ax_m.legend()
    ax_m.grid(True)

    plt.tight_layout()
    plt.show()
    return best_combo
def overlap_inverse(N,RA_P,DEC_P,FWHM_X_P,FWHM_Y_P,PA_P,RA_F,DEC_F,FWHM_X_F,FWHM_Y_F,PA_F): #PA vanno messi in radianti
    #Dapprima calcolo tutti i semiassi
    a_P=FWHM_X_P/2
    b_P=FWHM_Y_P/2
    a_F=FWHM_X_F/2
    b_F=FWHM_Y_F/2
    #Cerco le posizioni del centro dell'ellisse figlia rispetto alla padre
    dx = RA_F-RA_P
    dy= DEC_F-DEC_P
    x_rel=dx * np.cos(PA_P) + dy *np.sin(PA_P)
    y_rel= -dx *np.sin(PA_P)+dy *np.cos(PA_P)
    #Position Angle dell'ellisse figlia nel sistema di riferimento della padre
    PA_REL=PA_F-PA_P
    #Genero nell'ellisse padre N punti in maniera randomica. Per farlo passo in coordinate polari, dove l'ellisse è rappresentata da equaz.cerchio moltiplicate per rispettive semiassi
    u = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(u)
    x_rand = a_P * r * np.cos(theta)
    y_rand = b_P * r * np.sin(theta)
    #Conto quanti di essi sono nella padre e quanti in padre e figlia
    cont_PF = (( (x_rand-x_rel)* np.cos(PA_REL) + (y_rand-y_rel)*np.sin(PA_REL) )**2 / a_F**2 +  ( -(x_rand-x_rel)* np.sin(PA_REL) + (y_rand-y_rel)*np.cos(PA_REL) )**2 / b_F**2 <=1 ) .astype(float)
    over =  (np.sum(cont_PF)/N  ) * 100
    return over
def overlap(N,RA_P,DEC_P,FWHM_X_P,FWHM_Y_P,PA_P,RA_F,DEC_F,FWHM_X_F,FWHM_Y_F,PA_F): #PA vanno messi in radianti
    #Dapprima calcolo tutti i semiassi
    a_P=FWHM_X_P/2
    b_P=FWHM_Y_P/2
    a_F=FWHM_X_F/2
    b_F=FWHM_Y_F/2

    #Cerco le posizioni del centro della padre rispetto alla figlia
    dx = RA_P-RA_F
    dy= DEC_P-DEC_F

    x_rel=dx * np.cos(PA_F) + dy *np.sin(PA_F)
    y_rel= -dx *np.sin(PA_F)+dy *np.cos(PA_F)

    #Position Angle della padre nel sistema di riferimento della figlia
    PA_REL=PA_P-PA_F

    #Genero nell'ellisse figlia N punti in maniera randomica
    u = np.random.uniform(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    r = np.sqrt(u)

    x_rand = a_F * r * np.cos(theta)
    y_rand = b_F * r * np.sin(theta)

    #Conto quanti punti della figlia stanno anche nella padre
    cont_FP = (
        (
            ((x_rand-x_rel)* np.cos(PA_REL) + (y_rand-y_rel)*np.sin(PA_REL))**2 / a_P**2
            +
            (-(x_rand-x_rel)* np.sin(PA_REL) + (y_rand-y_rel)*np.cos(PA_REL))**2 / b_P**2
        ) <=1
    ).astype(float)

    over = (np.sum(cont_FP)/N) * 100

    return over
def stat_overlap(f_clump,f_id,ob_clump,ob_id,oc_clump,oc_id,cat_A,cat_B,cat_C,N=20,M=20,N_rand=10**5):

    overlap_m = np.empty((f_clump.shape[0], N, M), dtype=object)
    overlap_m_o = np.empty((ob_clump.shape[0], M), dtype=object)

    for i in range(f_clump.shape[0]):
         
        print(f"\rCaricamento non orfani: {100*(i+1)/f_clump.shape[0]:.1f}%", end="")
        for j in range( np.count_nonzero(f_clump[i,:,0]!=None)- 1 ): #uno in meno perchè parto da zero, un altro in meno perchè in prima riga ho padre
            
            idx = cat_B.index[(cat_B["CLUMP"].astype(str) == f_clump[i][j+1][0]) & (cat_B["ID"].astype(str) == f_id[i][j+1][0])][0] # trovo numero riga dataframe della sorgente figlia
            
            overlap_m[i][j+1][0] = overlap(
                N_rand,
                deg_to_arcsec(cat_A.loc[i,'RA']),
                deg_to_arcsec(cat_A.loc[i,'DEC']),
                cat_A.loc[i,'FWHM_X'],
                cat_A.loc[i,'FWHM_Y'],
                deg_to_rad(cat_A.loc[i,'PA']),
                deg_to_arcsec(cat_B.loc[idx,'RA']),
                deg_to_arcsec(cat_B.loc[idx,'DEC']),
                cat_B.loc[idx,'FWHM_X'],
                cat_B.loc[idx,'FWHM_Y'],
                deg_to_rad(cat_B.loc[idx,'PA'])
            )

            for k in range( np.count_nonzero(f_clump[i,j+1,:]!=None)- 1):
                
                idx_2 = cat_C.index[ (cat_C["CLUMP"].astype(str) == f_clump[i][j+1][k+1]) &  (cat_C["ID"].astype(str) == f_id[i][j+1][k+1])][0]
                
                overlap_m[i][j+1][k+1] = overlap(
                    N_rand,
                    deg_to_arcsec(cat_B.loc[idx,'RA']),
                    deg_to_arcsec(cat_B.loc[idx,'DEC']),
                    cat_B.loc[idx,'FWHM_X'],
                    cat_B.loc[idx,'FWHM_Y'],
                    deg_to_rad(cat_B.loc[idx,'PA']),
                    deg_to_arcsec(cat_C.loc[idx_2,'RA']),
                    deg_to_arcsec(cat_C.loc[idx_2,'DEC']),
                    cat_C.loc[idx_2,'FWHM_X'],
                    cat_C.loc[idx_2,'FWHM_Y'],
                    deg_to_rad(cat_C.loc[idx_2,'PA'])
                )
    for i in range(ob_clump.shape[0]):
        print(f"\rCaricamento orfani: {100*(i+1)/ob_clump.shape[0]:.1f}%", end="")
        idx_p = cat_B.index[(cat_B["CLUMP"].astype(str) == ob_clump[i][0]) & (cat_B["ID"].astype(str) == ob_id[i][0])][0]
        for j in range( np.count_nonzero(ob_clump[i,:]!=None) - 1 ):
            idx_f = cat_C.index[(cat_C["CLUMP"].astype(str) == ob_clump[i][j+1]) & (cat_C["ID"].astype(str) == ob_id[i][j+1])][0]
            overlap_m_o[i][j+1] = overlap(
                N_rand,
                deg_to_arcsec(cat_B.loc[idx_p,'RA']),
                deg_to_arcsec(cat_B.loc[idx_p,'DEC']),
                cat_B.loc[idx_p,'FWHM_X'],
                cat_B.loc[idx_p,'FWHM_Y'],
                deg_to_rad(cat_B.loc[idx_p,'PA']),
                deg_to_arcsec(cat_C.loc[idx_f,'RA']),
                deg_to_arcsec(cat_C.loc[idx_f,'DEC']),
                cat_C.loc[idx_f,'FWHM_X'],
                cat_C.loc[idx_f,'FWHM_Y'],
                deg_to_rad(cat_C.loc[idx_f,'PA'])
            )
    return overlap_m,overlap_m_o
