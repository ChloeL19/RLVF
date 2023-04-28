Require Import Coq.Lists.List.
Require Import Coq.Reals.Reals.
Require Import Coq.Init.Datatypes.

Open Scope list_scope.
Import ListNotations.

Fixpoint sum_list (l: list R) : R :=
  match l with
  | [] => 0
  | x :: xs => x + sum_list xs
  end.

Lemma example_sum_list_1: sum_list [1; 2; 3] = 6.
Proof. simpl. rewrite Rplus_comm. reflexivity. Qed.

Lemma example_sum_list_2: sum_list [1.5; 2.5; 3.0] = 7.0.
Proof. simpl. unfold Rdiv. rewrite Rmult_plus_distr_r. rewrite <- Rplus_assoc.
  rewrite Rplus_comm. reflexivity. Qed.

Lemma example_sum_list_3: sum_list ([]: list R) = 0.
Proof. simpl. reflexivity. Qed.