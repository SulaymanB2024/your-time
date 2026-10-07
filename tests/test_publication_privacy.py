import pytest

from publication_privacy import public_text

EMPTY = {'version': 1, 'replacements': []}


@pytest.mark.parametrize('body', [('/'.join(['', 'Users', 'private-person', 'source'])).encode(),
                                 b'person@' + b'personal-mail.net'])
def test_unconfigured_personal_information_blocks_publication(body):
    with pytest.raises(ValueError, match='Public source contains'):
        public_text(body, EMPTY)


def test_explicit_word_redaction_preserves_intentionally_public_account_name():
    policy = {'version': 1, 'replacements': [
        {'from': 'Private', 'to': 'yourtime', 'mode': 'word'}]}
    assert public_text(b'com.private.worker PrivateAccount2024/your-time', policy) == b'com.yourtime.worker PrivateAccount2024/your-time'


def test_synthetic_contacts_and_public_github_attribution_remain_valid():
    body = b'person@example.invalid owner@users.noreply.github.com noreply@github.com'
    assert public_text(body, EMPTY) == body


def test_invalid_policy_and_self_reintroducing_redaction_refuse_export():
    with pytest.raises(ValueError, match='Invalid publication'):
        public_text(b'source', {'version': 1, 'replacements': [{'from': '', 'to': 'x', 'mode': 'word'}]})
    with pytest.raises(ValueError, match='remains'):
        public_text(b'private-person', {'version': 1, 'replacements': [
            {'from': 'private-person', 'to': 'private-person', 'mode': 'word'}]})


@pytest.mark.parametrize('prefix', ['file://', 'file://localhost', ''])
def test_home_paths_in_file_uris_and_plain_text_refuse_export(prefix):
    body = (prefix + '/'.join(['', 'Users', 'unconfigured-person', 'file'])).encode()
    with pytest.raises(ValueError, match='personal home path'):
        public_text(body, EMPTY)


def test_encoded_and_windows_home_paths_refuse_export():
    path = '/'.join(['', 'Users', 'unconfigured-person', 'file']).encode()
    for body in (path.replace(b'/', b'%2F'), path.replace(b'/', b'\\/'),
                 b'C:' + bytes([92]) + b'Users' + bytes([92]) + b'unconfigured-person'):
        with pytest.raises(ValueError, match='personal home path'):
            public_text(body, EMPTY)
