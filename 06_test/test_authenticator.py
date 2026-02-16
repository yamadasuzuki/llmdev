import pytest
from authenticator import Authenticator


@pytest.fixture
def auth():
    """テスト用のAuthenticatorインスタンスを作成するフィクスチャ"""
    return Authenticator()

# register() メソッドで、ユーザーが正しく登録されるか
def test_register_new_user(auth):
    # ユーザー登録を実行
    auth.register("testuser", "password123")
    
    # ユーザーが正しく登録されたことを確認
    assert auth.users["testuser"] == "password123"
    assert "testuser" in auth.users

# register() メソッドで、すでに存在するユーザー名で登録を試みた場合に、エラーメッセージが出力されるか
def test_register_duplicate_user(auth):
    # 最初にユーザーを登録
    auth.register("testuser", "password123")
    
    # 同じユーザー名で再登録を試み、ValueError例外が発生することを確認
    with pytest.raises(ValueError) as exc_info:
        auth.register("testuser", "newpassword")
    
    # エラーメッセージが正しいことを確認
    assert str(exc_info.value) == "エラー: ユーザーは既に存在します。"

# login()メソッドで、正しいユーザー名とパスワードでログインできるか
def test_login_success(auth):
    # ユーザーを登録
    auth.register("testuser", "password123")
    
    # ログインを実行
    result = auth.login("testuser", "password123")
    
    # ログイン成功メッセージが返されることを確認
    assert result == "ログイン成功"

# login() メソッドで、誤ったパスワードでエラーが出るか
def test_login_wrong_password(auth):
    # ユーザーを登録
    auth.register("testuser", "password123")
    
    # 間違ったパスワードでログインを試み、ValueError例外が発生することを確認
    with pytest.raises(ValueError) as exc_info:
        auth.login("testuser", "wrongpassword")
    
    # エラーメッセージが正しいことを確認
    assert str(exc_info.value) == "エラー: ユーザー名またはパスワードが正しくありません。"