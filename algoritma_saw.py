import pandas as pd

class SAW:
    def __init__(self, df, weights, types):
        """
        df      : DataFrame berisi alternatif (baris) & kriteria (kolom)
        weights : bobot masing-masing kriteria (list)
        types   : jenis kriteria (list) -> "benefit" atau "cost"
        """
        self.df = df.copy()
        self.weights = weights
        self.types = types
        self.normalized = None
        self.scores = None

    def normalize(self):
        """
        Normalisasi matriks berdasarkan tipe kriteria.
        Benefit  -> dibagi max
        Cost     -> min dibagi nilai
        """
        norm_df = self.df.copy()

        for i, col in enumerate(norm_df.columns):
            if self.types[i] == "benefit":
                max_val = norm_df[col].max()
                norm_df[col] = norm_df[col] / max_val
            elif self.types[i] == "cost":
                min_val = norm_df[col].min()
                norm_df[col] = min_val / norm_df[col]
            else:
                raise ValueError("Tipe kriteria harus 'benefit' atau 'cost'.")

        self.normalized = norm_df
        return norm_df

    def calculate_scores(self):
        """
        Hasil akhir SAW = sum(normalized * weight)
        """
        if self.normalized is None:
            self.normalize()

        score = self.normalized.mul(self.weights, axis=1).sum(axis=1)
        self.scores = score
        return score

    def ranking(self):
        """
        Mengembalikan ranking dari nilai tertinggi ke rendah
        """
        if self.scores is None:
            self.calculate_scores()

        return self.scores.sort_values(ascending=False)
