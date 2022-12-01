

class StoredVal<T> {
    key: string
    _storage: T
    _convertForSave: ((storage: T) => any) | null = null
    _convertForLoad: ((jsonVal: any) => T) | null = null

    constructor(key: string, defaultVal: T, 
                convertForSave: ((storage: T) => any) | null = null,
                convertForLoad: ((jsonVal: any) => T) | null = null) {
        this.key = key
        this._convertForSave = convertForSave
        this._convertForLoad = convertForLoad
        var jsonVal = localStorage.getItem(key)
        console.log(`jsonVal ${jsonVal}`)
        if (jsonVal != null) {
            this._storage = JSON.parse(jsonVal)["data"]
            console.log(`now storage ${this._storage}`)
            if (this._storage == undefined) {
                this._storage = defaultVal
                console.log(`that was undefined`)
            }
            else if (this._convertForLoad != null) {
                console.log(`before convert ${this._storage}`)
                this._storage = this._convertForLoad(this._storage)
                console.log(`after convert  ${this._storage}`)
            }
        }
        else {
            this._storage = defaultVal
        }
    }

    get(): T {
        return this._storage
    }

    save() {
        var toStore: any
        if (this._convertForSave != undefined) {
            toStore = this._convertForSave(this._storage)
        }
        else {
            toStore = this._storage
        }
        localStorage.setItem(this.key, JSON.stringify({'data': toStore}))
    }
    
}

export { StoredVal }
