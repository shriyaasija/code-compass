import json

reasons = [
    "atomic() and Atomic represent the core transaction APIs, while BaseDatabaseWrapper handles the database connection and transaction state.",
    "BaseHandler is responsible for the overall request/response flow including middleware execution, while MiddlewareMixin is the base class for old-style middleware components.",
    "AbstractBaseUser and PermissionsMixin are the foundational models for custom user definitions, managed by BaseUserManager.",
    "ExtendsNode and BlockNode implement the core inheritance and block overriding mechanism in Django's template engine.",
    "SessionMiddleware integrates sessions into the request flow, while SessionBase and SessionStore implement the actual session storage logic.",
    "URLResolver and URLPattern handle the core URL pattern matching and resolution, tied together by the _path function for definition.",
    "BaseCommand provides the foundational class for all management commands, relying on CommandParser for handling CLI arguments.",
    "RequestContext manages the template context data while auth is the primary context processor for user-related variables.",
    "BaseForm implements the central form validation logic, raising ValidationError when data checks fail.",
    "QuerySet is the primary API for database queries, which are translated to SQL by the SQLCompiler and Query classes.",
    "StaticFilesStorage defines how static files are saved, while BaseFinder handles locating them across the filesystem.",
    "SecurityMiddleware enforces HTTP security headers, and CsrfViewMiddleware implements the CSRF protection mechanism.",
    "Apps maintains the registry of installed applications, and AppConfig configures the metadata for individual apps.",
    "The serialize function is the main entry point for serialization, delegating to the base Serializer class for implementation.",
    "BaseCache defines the abstract caching interface, while CacheHandler manages the active cache backends.",
    "Engine is the core evaluating context for templates, which are parsed from strings by the Parser class.",
    "BaseBackend defines the required interface for custom authentication, with ModelBackend providing the default implementation.",
    "Settings handles the configuration loading and attribute access, with LazySettings delaying evaluation until needed."
]

with open('devquery_bench/annotations_django__django.json', 'r') as f:
    data = json.load(f)

for i, item in enumerate(data):
    item['reason'] = reasons[i]

with open('devquery_bench/annotations_django__django.json', 'w') as f:
    json.dump(data, f, indent=2)
