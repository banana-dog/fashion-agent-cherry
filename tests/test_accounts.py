"""Accounts, and the fact that a wardrobe belongs to one person.

Every visitor used to arrive as `demo-user`, which meant the same wardrobe, the
same photographs and the same measurements for everybody who ever opened the
page. These tests are mostly about that not being possible any more.
"""

import pytest

from fashion_agent.accounts import (
    AccountError,
    Accounts,
    check_passphrase,
    check_passphrase_length,
    clean_login,
    hash_passphrase,
)

PHRASE = "parol-dlinnaya-1"


@pytest.fixture
def accounts(tmp_path):
    return Accounts(tmp_path / "accounts.sqlite3")


class TestPassphrases:
    def test_a_passphrase_is_not_stored(self, accounts):
        account = accounts.register("mira", PHRASE)

        with accounts._lock:
            row = accounts._connection().execute(
                "SELECT salt, secret FROM accounts WHERE user_id = ?", (account.user_id,)
            ).fetchone()

        assert PHRASE.encode() not in bytes(row["secret"])
        assert PHRASE.encode() not in bytes(row["salt"])

    def test_two_accounts_get_different_salts(self):
        first, first_secret = hash_passphrase(PHRASE)
        second, second_secret = hash_passphrase(PHRASE)

        assert first != second
        assert first_secret != second_secret

    def test_the_right_passphrase_is_recognised(self):
        salt, secret = hash_passphrase(PHRASE)

        assert check_passphrase(PHRASE, salt, secret) is True

    def test_a_wrong_passphrase_is_not(self):
        salt, secret = hash_passphrase(PHRASE)

        assert check_passphrase("other", salt, secret) is False

    def test_a_passphrase_too_long_to_hash_is_wrong_rather_than_fatal(self):
        salt, secret = hash_passphrase(PHRASE)

        assert check_passphrase("x" * 100_000, salt, secret) is False

    def test_a_short_passphrase_is_refused(self):
        with pytest.raises(AccountError):
            check_passphrase_length("short")

    def test_an_empty_passphrase_is_refused(self):
        with pytest.raises(AccountError):
            check_passphrase_length("")

    def test_a_login_is_normalised(self):
        assert clean_login("  Mira  ") == "mira"

    def test_a_name_that_is_too_short_is_refused(self):
        with pytest.raises(AccountError):
            clean_login("ab")


class TestRegistration:
    def test_a_new_visitor_gets_an_account_of_their_own(self, accounts):
        first = accounts.anonymous()
        second = accounts.anonymous()

        assert first.user_id != second.user_id
        assert first.anonymous is True

    def test_a_visitor_id_is_not_guessable(self, accounts):
        user_id = accounts.anonymous().user_id

        # A short, memorable id is a name in a directory.
        assert len(user_id) >= 16
        assert not user_id.isalpha()

    def test_registration_adopts_the_visitors_data(self, accounts):
        """A wardrobe built before registering must survive registering."""
        visitor = accounts.anonymous()

        account = accounts.register("mira", PHRASE, take_over_from=visitor.user_id)

        assert account.user_id == visitor.user_id
        assert account.anonymous is False

    def test_registration_works_without_anything_to_adopt(self, accounts):
        account = accounts.register("mira", PHRASE)

        assert accounts.account_for(account.user_id).login == "mira"

    def test_a_taken_name_is_refused(self, accounts):
        accounts.register("mira", PHRASE)

        with pytest.raises(AccountError, match="занято"):
            accounts.register("mira", "other-phrase")

    def test_a_registered_account_cannot_be_adopted_again(self, accounts):
        first = accounts.register("mira", PHRASE)
        second = accounts.anonymous()

        with pytest.raises(AccountError, match="уже зарегистрирован"):
            accounts.register("other", PHRASE, take_over_from=first.user_id)

        assert accounts.account_for(second.user_id).anonymous is True

    def test_a_bad_registration_changes_nothing(self, accounts):
        visitor = accounts.anonymous()

        with pytest.raises(AccountError):
            accounts.register("ab", PHRASE, take_over_from=visitor.user_id)

        assert accounts.account_for(visitor.user_id).anonymous is True


class TestSignIn:
    def test_the_right_credentials_get_in(self, accounts):
        accounts.register("mira", PHRASE)

        assert accounts.sign_in("mira", PHRASE).user_id

    def test_a_wrong_passphrase_is_refused(self, accounts):
        accounts.register("mira", PHRASE)

        with pytest.raises(AccountError):
            accounts.sign_in("mira", "not-the-phrase")

    def test_a_name_that_does_not_exist_says_the_same_thing(self, accounts):
        """A different message would tell a stranger who has an account here."""
        accounts.register("mira", PHRASE)

        with pytest.raises(AccountError) as wrong_phrase:
            accounts.sign_in("mira", "not-the-phrase")

        with pytest.raises(AccountError) as no_such_name:
            accounts.sign_in("nobody", PHRASE)

        assert str(wrong_phrase.value) == str(no_such_name.value)

    def test_the_name_is_matched_case_insensitively(self, accounts):
        account = accounts.register("Mira", PHRASE)

        assert accounts.sign_in("mira", PHRASE).user_id == account.user_id

    def test_an_empty_passphrase_is_refused(self, accounts):
        accounts.register("mira", PHRASE)

        with pytest.raises(AccountError):
            accounts.sign_in("mira", "")


class TestSessions:
    def test_a_token_finds_its_owner(self, accounts):
        account = accounts.register("mira", PHRASE)
        token = accounts.open_session(account.user_id)

        assert accounts.user_for_token(token) == account.user_id

    def test_tokens_are_not_predictable_from_each_other(self, accounts):
        account = accounts.register("mira", PHRASE)

        assert accounts.open_session(account.user_id) != accounts.open_session(
            account.user_id
        )

    def test_no_token_means_no_owner(self, accounts):
        assert accounts.user_for_token(None) is None
        assert accounts.user_for_token("") is None

    def test_a_made_up_token_is_not_looked_up(self, accounts):
        """A guess must not become a query, let alone a match."""
        for token in ("../../../etc/passwd", "' OR 1=1 --", "a" * 500, "../../"):
            assert accounts.user_for_token(token) is None

    def test_an_unknown_token_finds_nobody(self, accounts):
        assert accounts.user_for_token("x" * 43) is None

    def test_signing_out_closes_that_one_device(self, accounts):
        account = accounts.register("mira", PHRASE)
        here = accounts.open_session(account.user_id)
        there = accounts.open_session(account.user_id)

        accounts.close_session(here)

        assert accounts.user_for_token(here) is None
        assert accounts.user_for_token(there) == account.user_id

    def test_signing_out_everywhere_closes_all_of_them(self, accounts):
        account = accounts.register("mira", PHRASE)
        accounts.open_session(account.user_id)
        accounts.open_session(account.user_id)

        assert accounts.close_sessions_of(account.user_id) == 2

    def test_a_session_survives_a_new_process(self, tmp_path):
        """Otherwise a restart signs everybody out and they lose their place."""
        path = tmp_path / "accounts.sqlite3"
        first = Accounts(path)
        account = first.register("mira", PHRASE)
        token = first.open_session(account.user_id)

        assert Accounts(path).user_for_token(token) == account.user_id

    def test_an_expired_session_is_refused(self, accounts, monkeypatch):
        import datetime

        account = accounts.register("mira", PHRASE)
        token = accounts.open_session(account.user_id)

        # A week is how long a session lasts, so the clock has to pass that.
        later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=8)
        monkeypatch.setattr(
            "fashion_agent.accounts.utc_now", lambda: later
        )

        assert accounts.user_for_token(token) is None
        assert accounts.session_count(account.user_id) == 0

    def test_expired_sessions_can_be_swept(self, accounts, monkeypatch):
        import datetime

        account = accounts.register("mira", PHRASE)
        accounts.open_session(account.user_id)
        # A week is how long a session lasts, so the clock has to pass that.
        later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=8)
        monkeypatch.setattr("fashion_agent.accounts.utc_now", lambda: later)

        assert accounts.sweep() == 1

    def test_forgetting_an_account_takes_its_sessions_with_it(self, accounts):
        account = accounts.register("mira", PHRASE)
        token = accounts.open_session(account.user_id)

        accounts.forget(account.user_id)

        assert accounts.account_for(account.user_id) is None
        assert accounts.user_for_token(token) is None
